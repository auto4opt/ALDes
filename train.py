import os
import time
from torch.optim import Adam

from data import *
from models.model.transformer import Transformer
from util.bleu import idx_to_word, get_bleu

import matlab.engine
import matlab
from matlab_setting import *
from EWC import *
from pflacco_feature import cal_feature, transform
from util.my_util import *

useEWC = False
EWC_ = None

eng = matlab.engine.start_matlab()
# matlab root dir
work_path = os.getcwd() + "\\matlab"
eng.cd(work_path)

logs = my_log()


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.kaiming_uniform(m.weight.data)


def get_model():
    if model_type == 0:
        model = Transformer(src_pad_idx=src_pad_idx,
                            trg_pad_idx=trg_pad_idx,
                            trg_sos_idx=trg_sos_idx,
                            d_model=d_model,
                            enc_voc_size=total_index,  # enc_voc_size,
                            dec_voc_size=total_index,  # voc_size,
                            max_len=max_len,
                            ffn_hidden=ffn_hidden,
                            n_head=n_heads,
                            n_layers=n_layers,
                            drop_prob=drop_prob,
                            device=device).to(device)

        print(f'The model has {count_parameters(model):,} trainable parameters')
        model.apply(initialize_weights)
        optimizer = Adam(params=model.parameters(),
                         lr=init_lr,
                         weight_decay=weight_decay,
                         eps=adam_eps)
    else:
        pass
    return model, optimizer


def get_performance(action, problem_id, eval=0):
    # call matlab funtion get performance
    mean_performances = []
    performances = []
    if eval == 1:
        action = action[0:3]
    for j in range(action.shape[0]):
        # get performance and reward
        alg_list = action[j, :].reshape(-1).tolist()

        # delet first one "Begin"
        alg_list.pop(0)
        # delet "End"
        while (alg_list[len(alg_list) - 1] == end_index):
            alg_list.pop()  # delet last one
        # print(alg_list)
        alg = matlab.double(initializer=alg_list)
        if eval == 0:
            instances = matlab.double([1, 2, 3])
        else:
            instances = matlab.double([4])
        performance = eng.get_perf(alg, problem_id, instances, eval, nargout=1)

        if eval == 0:
            performance = np.array(performance)
            performance = performance[0:3]
        else:
            performance = np.array(performance)
            performance = performance[0]
        # performance = np.array(performance) #first row is instanceTrain , second is instanceTest, do not use Test
        mean_performance = performance.mean()
        mean_performances.append(mean_performance)
        performances.append(performance)
        if eval == 1:
            logs.write_log("problem_" + (problem_id).__str__() + " result:\n" + str(performance))
            logs.write_log("problem_" + (problem_id).__str__() + " result mean:\n" + str(mean_performance))

    return mean_performances, performances


def PPO(model, optimizer, clip, total_epoch, ppo_epoch, baseline, clip_coef, src, trg, att_src, problem_id):
    ewc_loss = None
    global EWC_
    action_total_list = []
    log_performances = []
    # ppo
    for i in range(total_epoch):
        since = time.time()
        with torch.no_grad():
            action, action_p, action_log_p = model(src, trg, att_src)
            action_log_p = torch.squeeze(action_log_p, 2)
            logs.write_log("action in train: \n" + str(action[0:5]))
            logs.write_log("action_p in train: \n " + str(torch.squeeze(action_p[0:5], 2)))
            action_total_list += action.tolist()

        mean_performances, performances = get_performance(action, problem_id)
        log_performances.append(performances)
        logs.write_log("performances in train: \n " + str(performances[0:5]))
        mean_performances = torch.tensor(mean_performances)
        cost = mean_performances.to(device)
        if baseline is None:
            baseline = cost.mean()
        else:
            baseline = 0.8 * baseline + 0.2 * cost.mean()
        baseline = baseline.detach()

        # ppo update
        for j in range(ppo_epoch):

            _, new_action_p, new_action_log_p = model(src, trg, att_src, action)
            new_action_log_p = torch.squeeze(new_action_log_p, 2)
            logratio = new_action_log_p.sum(1) - action_log_p.sum(1)
            ratio = logratio.exp()

            pg_loss1 = (cost - baseline) * ratio
            pg_loss2 = (cost - baseline) * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            loss = torch.max(pg_loss1, pg_loss2).mean()

            if EWC_ is not None:
                ewc_loss = EWC_.penalty(model)
                loss += ewc_loss
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        if ewc_loss is not None:
            logs.write_log(('step :', round((i / total_epoch) * 100, 2),
                            '% , ewc_loss :', ewc_loss.item()).__str__())
            print('step :', round((i / total_epoch) * 100, 2),
                  '% , ewc_loss :', ewc_loss.item())

        time_elapsed = time.time() - since

        print('step :', round((i / total_epoch) * 100, 2),
              '% , loss :', loss.item(),
              ', cost_mean :', cost.mean().item(),
              ', baseline :', baseline.item(),
              ',Training complete in {:.0f}m {:.0f}s'.format(
                  time_elapsed // 60, time_elapsed % 60)
              )
        logs.write_log(('step :', round((i / total_epoch) * 100, 2),
                        '% , loss :', loss.item(),
                        ', cost_mean :', cost.mean().item(),
                        ', baseline :', baseline.item(),
                        ',Training complete in {:.0f}m {:.0f}s'.format(
                            time_elapsed // 60, time_elapsed % 60)
                        ).__str__())

    logs.dump_log(log_performances)

    return action


def PPO_get_ewc(model, optimizer, clip, total_epoch, ppo_epoch, baseline, clip_coef, src, trg, att_src, problem_id):
    # ppo
    for i in range(1):
        since = time.time()
        with torch.no_grad():
            action, action_p, action_log_p = model(src, trg, att_src)
            action_log_p = torch.squeeze(action_log_p, 2)

        mean_performances, performances = get_performance(action, problem_id)
        mean_performances = torch.tensor(mean_performances)
        cost = mean_performances.to(device)
        if baseline is None:
            baseline = cost.mean()
        else:
            baseline = 0.8 * baseline + 0.2 * cost.mean()
        baseline = baseline.detach()

        # ppo update
        for j in range(1):
            _, new_action_p, new_action_log_p = model(src, trg, att_src, action)
            new_action_log_p = torch.squeeze(new_action_log_p, 2)
            logratio = new_action_log_p.sum(1) - action_log_p.sum(1)
            ratio = logratio.exp()

            pg_loss1 = (cost - baseline) * ratio
            pg_loss2 = (cost - baseline) * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            loss = torch.max(pg_loss1, pg_loss2).mean()

            optimizer.zero_grad()
            loss.backward()
            EWC_.update_diag_fisher(model)
    return action


def train(model, optimizer, clip, problem_id, src, trg, att_src):
    model.train()
    baseline = None
    clip_coef = 0.2
    if train_type == 0:
        action = PPO(model, optimizer, clip, total_epoch, ppo_epoch, baseline, clip_coef, src, trg, att_src, problem_id)

    return action


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    batch_bleu = []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:, :-1])
            output_reshape = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            loss = criterion(output_reshape, trg)
            epoch_loss += loss.item()

            total_bleu = []
            for j in range(batch_size):
                try:
                    trg_words = idx_to_word(batch.trg[j], loader.target.vocab)
                    output_words = output[j].max(dim=1)[1]
                    output_words = idx_to_word(output_words, loader.target.vocab)
                    bleu = get_bleu(hypotheses=output_words.split(), reference=trg_words.split())
                    total_bleu.append(bleu)
                except:
                    pass

            total_bleu = sum(total_bleu) / len(total_bleu)
            batch_bleu.append(total_bleu)

    batch_bleu = sum(batch_bleu) / len(batch_bleu)
    return epoch_loss / len(iterator), batch_bleu


def train_separately():
    print("Train separately")
    logs.write_log("Train separately")
    problem_set = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
    problem_nums = 0
    input_src = []
    for problem_id in problem_set:
        logs.problem_id = problem_id
        print("Train problem_" + problem_id.__str__())
        logs.write_log("Train problem_" + problem_id.__str__())

        model, optimizer = get_model()

        src = torch.randn(10, d_model).to(device)  # 512->30
        trg = torch.tensor([begin_index]).to(device)
        trg = trg.unsqueeze(dim=0)
        # train with batch
        src = src.repeat(batch_size_src, 1, 1)
        trg = trg.repeat(batch_size_src, 1)
        att_src = torch.range(0, 26, 1, dtype=torch.int).to(device)
        att_src = att_src.unsqueeze(dim=0)
        att_src = att_src.repeat(batch_size_src, 1)

        input_src.append(src)
        logs.write_log("model input :\n" + str(src))

        action = train(model, optimizer, clip, problem_id, src, trg, att_src)

        logs.write_log("train over action: \n" + str(action))

        # get_performance(action,problem_id,eval=1)


def train_in_one():
    print("Train In One")
    logs.write_log("Train In One")
    input_src = {}
    model, optimizer = get_model()
    global EWC_
    problem_set = [1, 2, 11, 18, 19, 22, 23]
    problem_nums = 0
    feature = transform()
    for problem_id in problem_set:
        src = torch.from_numpy(feature[problem_id - 1][0:d_model].reshape(1, d_model)).to(torch.float32).to(device)
        src = src.repeat(batch_size_src, 1, 1)

        input_src[problem_id] = src
    for problem_id in problem_set:
        logs.problem_id = problem_id
        problem_nums += 1
        print("Train problem_" + problem_id.__str__())
        logs.write_log("Train problem_" + problem_id.__str__())

        trg = torch.tensor([begin_index]).to(device)
        trg = trg.unsqueeze(dim=0)
        # train with batch
        trg = trg.repeat(batch_size_src, 1)
        att_src = torch.range(0, 26, 1, dtype=torch.int).to(device)
        att_src = att_src.unsqueeze(dim=0)
        att_src = att_src.repeat(batch_size_src, 1)

        action = train(model, optimizer, clip, problem_id, input_src[problem_id], trg, att_src)

        # logs.write_log("train over action: \n"+str(action))
        if useEWC is True:
            print("Cal EWC______________________")
            EWC_ = EWC(model)
            for i in problem_set:
                temp_src = input_src[i]
                PPO_get_ewc(model, optimizer, clip, total_epoch, ppo_epoch, None, 0.2, temp_src, trg, att_src, i)
                if i == problem_id: break
            for key in EWC_._precision_matrices:
                EWC_._precision_matrices[key] = EWC_._precision_matrices[key] / (problem_nums)

        torch.save(model.state_dict(), 'logs/' + "train_in_one/" + problem_id.__str__() + '.pt')

        logs.write_log("EWC TEST : train over problem : " + (problem_id).__str__())
        for key, value in input_src.items():
            old_action, action_p, action_log_p = model(input_src[key], trg, att_src, reference=True)
            logs.write_log("problem_" + key.__str__() + " action:\n" + str(old_action))
            get_performance(old_action[0:1], key, eval=1)


def test():
    model, optimizer = get_model()
    model.load_state_dict(
        torch.load(r'D:\01Code\transformer-rl-3\logs\Transformer_PPO\100_32\1_11_17_12_11\train_in_one.pt'))
    input_src = []
    problem_set = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    problem_nums = 0
    for problem_id in problem_set:
        src = torch.randn(30, d_model).to(device)  # 512->30
        trg = torch.tensor([begin_index]).to(device)
        trg = trg.unsqueeze(dim=0)
        # train with batch
        src = src.repeat(batch_size_src, 1, 1)
        trg = trg.repeat(batch_size_src, 1)

        att_src = torch.range(0, 26, 1, dtype=torch.int).to(device)
        att_src = att_src.unsqueeze(dim=0)
        att_src = att_src.repeat(batch_size_src, 1)
        input_src.append(src)

        with torch.no_grad():
            action, action_p, action_log_p = model(src, trg, att_src)
            action_log_p = torch.squeeze(action_log_p, 2)

    mean_performances, performances = get_performance(action, problem_id)
    mean_performances = torch.tensor(mean_performances)


if __name__ == '__main__':
    for seed in range(1, 6, 1):
        logs.seed = seed
        logs.write_log('seed is {}'.format(seed))
        seed_torch(seed)
        #train_in_one()
        train_separately()
