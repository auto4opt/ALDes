import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from models.blocks.decoder_layer import DecoderLayer
from models.embedding.transformer_embedding import TransformerEmbedding
from torch.distributions.normal import Normal
from numpy import*
from torch.nn import Parameter
from matlab_setting import *
use_Point = False


class GatingMechanism(torch.nn.Module):
    def __init__(self, d_input, bg=0.1):
        super(GatingMechanism, self).__init__()
        self.Wr = torch.nn.Linear(d_input, d_input)
        self.Ur = torch.nn.Linear(d_input, d_input)
        self.Wz = torch.nn.Linear(d_input, d_input)
        self.Uz = torch.nn.Linear(d_input, d_input)
        self.Wg = torch.nn.Linear(d_input, d_input)
        self.Ug = torch.nn.Linear(d_input, d_input)
        self.bg = bg

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()

    def forward(self, x, y):
        r = self.sigmoid(self.Wr(y) + self.Ur(x))
        z = self.sigmoid(self.Wz(y) + self.Uz(x) - self.bg)
        h = self.tanh(self.Wg(y) + self.Ug(torch.mul(r, x)))
        g = torch.mul(1 - z, x) + torch.mul(z, h)
        return g
class Decoder(nn.Module):
    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,
                                        drop_prob=drop_prob,
                                        max_len=max_len,
                                        vocab_size=dec_voc_size,
                                        device=device)

        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax =nn.Softmax(dim=2)
        self.device = device
        self.temp = 1.0
        self.input_linear = nn.Linear(d_model, d_model)
        self.action_linear = nn.Conv1d(d_model, d_model, 1, 1)
        self.V = Parameter(torch.FloatTensor(d_model), requires_grad=True)
        self.tanh = nn.Tanh()
        self.attn = nn.Linear(d_model, d_model)

        # Initialize vector V
        nn.init.uniform(self.V, -1, 1)

        self.gate = GatingMechanism(d_model)

    def get_action(self, model_output, action = None):

        model_output = torch.squeeze(model_output)
        sdv = model_output[1::2]
        mean = model_output[0::2]
        # here use exp make sure its positive
        sdv = torch.exp(sdv)

        probs = Normal(mean, sdv)
        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy()

    def pointer(self, out, action_emb):
        out = self.input_linear(out).expand(-1,action_emb.size(1),-1)
        out = out.permute(0, 2, 1)
        action_emb = action_emb.permute(0,2,1)
        action_emb = self.action_linear(action_emb)
        # (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(action_emb.size(0), -1).unsqueeze(1)

        # (batch, seq_len)
        att = torch.bmm(V, self.tanh(out + action_emb)).squeeze(1)
        return att
    def forward(self, trg, enc_src, action_emb,trg_mask, src_mask,action = None,inference = False):

        action_p=[]
        action_log_p=[]
        operator_num = torch.zeros(trg.shape,dtype=torch.int64).to(self.device)
        need_para_num = torch.zeros(trg.shape,dtype=torch.int64).to(self.device)
        action_index = torch.zeros(trg.shape).to(self.device) # 'begin'
        last_opera_index =torch.zeros(trg.shape,dtype=torch.int64).to(self.device)
        last_normal_opera_index = torch.zeros(trg.shape,dtype=torch.int64).to(self.device)
        have_gs = torch.zeros(trg.shape, dtype=torch.int64).to(self.device)
        action_index[:,:]=begin_index
        last_opera_index[:,:]=begin_index
        finished = torch.zeros(trg.shape).to(self.device)
        all_finished = torch.ones(trg.shape).to(self.device)

        ppo_index = 1

        #for i in range(0,16):
        while(True):
            #code here : drop finished row
            #unfinished_index,last_opera_index,last_normal_opera_index, operator_num,need_para_num, unfinished_trg,unfinished_have_gs,unfinished_action_emb,unfinished_enc_src\
            #    = self.choose_unfinished(finished,have_gs,last_opera_index,last_normal_opera_index,operator_num,need_para_num,trg,action_emb,enc_src)

            input = self.emb(trg)

            #input = torch.cat((unfinished_enc_src, trg_emb), dim=1)

            for layer in self.layers:
                 input = layer(input,trg_mask = None)

            # pass to LM head
            output = self.linear(input)
            output = output[:,-1:,:]
            #mask = self.get_mask_mat(unfinished_trg, action_index, last_opera_index,last_normal_opera_index, operator_num, need_para_num, unfinished_have_gs, finished).to(self.device)
            mask = self.get_mask(trg)
            mask = torch.unsqueeze(mask,1)
            output = output.masked_fill(mask == 0, -math.inf)
            # See https://discuss.pytorch.org/t/bad-behavior-of-multinomial-function/10232
            _log_p = torch.log_softmax(output / self.temp, dim=-1)
            distribution = _log_p.exp()
            if torch.isnan(distribution).any() or torch.isinf(distribution).any():
                print("nan or inf")

            if action is None:
                action_index = distribution.squeeze(1).multinomial(1)
                if inference is True:action_index = torch.argmax(distribution.squeeze(1), dim=1).unsqueeze(1)
                while not mask.squeeze(1).gather(1, action_index).data.all():
                    print('Sampled bad values, resampling!')
                    action_index = output.multinomial(1).squeeze(1)
                    if inference is True: action_index = torch.argmax(distribution.squeeze(1), dim=1).unsqueeze(1)
            else:
                # action should be full rows
                unfinished_index = torch.where(finished == 0)[0]
                temp_action = torch.index_select(action, 0, unfinished_index)
                action_index = temp_action[:,ppo_index] # "begin" is first
                ppo_index += 1
                action_index = action_index.unsqueeze(1)

            # Get log_p corresponding to selected actions
            p = distribution[0, 0, action_index]
            log_p = _log_p.gather(2, action_index.unsqueeze(-1)).squeeze(-1)

            trg = torch.cat([trg, action_index], dim=1)
            action_p.append(p)
            action_log_p.append(log_p)
            if torch.all(action_index == end_index):
                break

        trg = self.check_pointer(trg)
        return trg, torch.stack(action_p,1),torch.stack(action_log_p,1)

    # if pointer point more than the alg operator length, fix it
    def check_pointer(self,cur_alg):
        dim0, dim1 = cur_alg.shape
        for i in range(dim0):
            cur_opera_num = 0
            cur_operas = []
            for action in cur_alg[i]:
                if operators_begin <= action <= operators_end:
                    cur_opera_num += 1
                    cur_operas.append(action)

            positions = torch.nonzero(torch.eq(cur_alg[i], fork_index))
            for position in positions:
                if cur_alg[i][position + 1] > para_begin + cur_opera_num - 1:
                    cur_alg[i][position + 1] = para_begin + cur_opera_num - 1

        return cur_alg

    def get_mask(self, cur_alg):

        dim0,dim1 = cur_alg.shape
        # 0 means action is masked
        mask = torch.zeros([dim0, voc_size], dtype=torch.double).to(self.device)
        for i in range(dim0):
            last_action = cur_alg[i][-1]

            # first action must be 'choose'
            if last_action == begin_index:
                mask[i][choose_begin:choose_end] = 1
                continue

            # last action is "end"
            if last_action == end_index:
                mask[i][end_index] = 1
                continue

            tmp = -1
            # last not para action is operator or ptr
            last_opera_or_ptrs = True
            last_opera = None
            last_ptrs = None
            while True:
                if operators_begin <= cur_alg[i][tmp] <= operators_end and last_opera is None:
                    last_opera = cur_alg[i][tmp]
                if ptrs_begin <= cur_alg[i][tmp] <= ptrs_end and last_ptrs is None:
                    last_ptrs = cur_alg[i][tmp]
                if (last_ptrs is not None and last_opera is not None) or cur_alg[i][tmp] == begin_index:
                    break
                tmp -= 1
            tmp = -1
            while True:
                if operators_begin <= cur_alg[i][tmp] <= operators_end:
                    last_opera_or_ptrs = True
                    break
                if ptrs_begin <= cur_alg[i][tmp] <= ptrs_end:
                    last_opera_or_ptrs = False
                    break
                tmp -= 1


            # calculate total operators of cur alg
            tmp = -1
            cur_opera_num = 0
            cur_operas = []
            for action in cur_alg[i]:
                if operators_begin <= action <= operators_end:
                    cur_opera_num += 1
                    cur_operas.append(action)

            operator_need_para = False
            need_operator_or_end = False
            # last action is operator
            if operators_begin <= last_action <= operators_end:
                if need_para_list[last_action] != 0:
                    mask[i][para_begin:para_end + 1] = 1
                    operator_need_para = True
                else:
                    # operator have no para, need pointer
                    if choose_begin <= last_action <= choose_end:
                        mask[i][forward_index] = 1
                        mask[i][fork_index] = 1
                    elif search_begin <= last_action <= search_end:
                        mask[i][ptrs_begin:ptrs_end + 1] = 1
                    elif update_begin <= last_action <= update_end:
                        mask[i][forward_index] = 1
                        #mask[i][iterate_index] = 1

            # last action is para
            elif para_begin <= last_action <= para_end:
                para_enough = False
                # para of operator
                if last_opera_or_ptrs:
                    if para_begin <= cur_alg[i][-2] <= para_end:
                        para_enough = True
                    else:
                        if need_para_list[last_opera] == 1:
                            para_enough = True
                    if para_enough:
                        # opear,para,need ptr
                        if choose_begin <= last_opera <= choose_end:
                            mask[i][forward_index] = 1
                            mask[i][fork_index] = 1
                        elif search_begin <= last_opera <= search_end:
                            mask[i][ptrs_begin:ptrs_end + 1] = 1
                        elif update_begin <= last_opera <= update_end:
                            mask[i][forward_index] = 1
                            mask[i][iterate_index] = 1
                    else:
                        mask[i][para_begin:para_end + 1] = 1
                        operator_need_para = True
                # para of ptrs
                else:
                    # ptrs most have one para, next is operator/end
                    need_operator_or_end = True

            # last action is pointer
            elif ptrs_begin <= last_action <= ptrs_end:
                if last_action == forward_index:
                    # forward have no condition
                    need_operator_or_end = True
                elif last_action == iterate_index:
                    # iterate only follow para in 1-5
                    mask[i][para_begin:para_begin + 5] = 1
                elif last_action == fork_index:
                    mask[i][para_begin + cur_opera_num + 1 : para_begin + max_operators] = 1
                    if para_begin + cur_opera_num + 1 >= max_operators :
                        mask[i][:] = 0
                        mask[i][para_begin + max_operators - 1] = 1

            if need_operator_or_end:
                if cur_opera_num == 1:
                    mask[i][search_begin:search_end + 1] = 1
                    # first search not be mu
                    mask[i][mu_begin:mu_end + 1] = 0
                elif 1 < cur_opera_num < max_operators - 2:
                    mask[i][search_begin:search_end + 1] = 1
                    mask[i][update_begin:update_end + 1] = 1
                elif cur_opera_num >= max_operators - 2:
                    mask[i] = 0
                    mask[i][update_begin:update_end + 1] = 1

                if update_begin <= last_opera <= update_end:
                    mask[i][:] = 0
                    mask[i][end_index] = 1

                # if last operator is crossover,next must be mutation
                if cross_begin <= last_opera <= cross_end:
                    mask[i][:] = 0
                    mask[i][mu_begin:mu_end + 1] = 1
                # mask already selected operators
                for select_opeartor in cur_operas:
                    mask[i][select_opeartor] = 0

            # here need mask global para or operator
            have_global_opera = False
            for j in range(len(cur_alg[i])):
                action = cur_alg[i][j]
                if operators_begin <= action <= operators_end:
                    if action in only_gs_opera_index:
                        have_global_opera = True
                    # no para search only be local
                    elif need_para_list[action] >= 1:
                        # all search most have one para, para>0.3 is GS
                        if j != len(cur_alg[i]) -1 and cur_alg[i][j+1] >= gs_para_begin:
                            have_global_opera = True
            if have_global_opera:
                mask[i][only_gs_opera_index] = 0
                if operator_need_para:
                    mask[i][para_begin:gs_para_begin] = 0

            if mask[i].sum() == 0:
                print("all mask ! ERROR")

        return mask









    def get_mask_mat(self, trg, action_index,last_opera_index,last_normal_opera_index, operator_num, need_para_num,have_gs,finished):

        dim0,dim1 = trg.shape

        last_is_prts = (last_opera_index >= ptrs_begin) * (last_opera_index <= ptrs_begin + ptrs_num - 1)
        last_normal_is_cross = (last_normal_opera_index<=cross_begin+cross_num-1) * (last_normal_opera_index>=cross_begin)

        # 0 is masked
        mask = torch.ones([dim0,voc_size],dtype=torch.double).to(self.device)

        # mask parameter
        mask[:,para_begin:para_begin+para_num] = torch.where(need_para_num!=0, mask[:,para_begin:para_begin+para_num],0.)

        # mask choose operator
        choose_cond = (operator_num == 0) * (need_para_num == 0)
        mask[:, choose_begin:choose_begin + choose_num] = torch.where(choose_cond, mask[:, choose_begin:choose_begin + choose_num], 0.)
        # mask update operator
        selet_cond = (operator_num == 1) * (need_para_num == 0)
        mask[:, update_begin:update_begin+update_num] = torch.where(selet_cond, mask[:, update_begin:update_begin+update_num], 0.)

        # mask search operator
        search_cond = (operator_num >= 2) * (need_para_num == 0)
        mask[:, search_begin:search_begin + search_num] = torch.where(search_cond, mask[:, search_begin:search_begin + search_num], 0.)
        # crossover must followed by mutation,if last operator is crossover, mask operators unless mutation
        cross_cond = last_normal_is_cross* (need_para_num == 0)
        # first mask all operator and unmask mutation
        mask[:, search_begin:search_begin + search_num] = torch.where(cross_cond, 0. ,mask[:, search_begin:search_begin + search_num])
        mask[:, mu_begin:mu_begin+mu_num] = torch.where(cross_cond,1.,mask[:, mu_begin:mu_begin+mu_num])

        #first search not be mutation
        first_search = (operator_num == 2)
        mask[:, mu_begin:mu_begin + mu_num] = torch.where(first_search, 0., mask[:, mu_begin:mu_begin + mu_num])

        # mask seleted operator, not contain ptrs
        for i in range(dim0):
            for j in range(dim1):
                opera = trg[i][j].item()
                if opera<=operators_num: mask[i,opera] = 0.

        # alg have search, choose, update, can end, add "End"
        mask[:,end_index]=0
        temp = torch.ones([dim0,total_index],dtype=torch.double).to(self.device)
        can_end = (operator_num >=4) * (need_para_num ==0) * (~last_normal_is_cross) *(last_is_prts)
        mask[:,end_index:end_index+1] = torch.where(can_end ,temp[:,end_index:end_index+1],mask[:,end_index:end_index+1])

        # last operator cant be cross
        last_opera = (operator_num >=8)*(need_para_num == 0)
        mask[:, cross_begin:cross_begin +cross_num] = torch.where(last_opera,0.,mask[:, cross_begin:cross_begin +cross_num])

        # force select ptrs after search opera and para
        need_ptrs = (need_para_num == 0) * (last_opera_index >= search_begin) * (last_opera_index <= search_begin + search_num - 1)
        need_ptr_mask = torch.zeros([dim0,total_index],dtype=torch.double).to(self.device)
        need_ptr_mask[:, ptrs_begin : ptrs_begin + ptrs_num] = 1
        mask[:,:] = torch.where(need_ptrs,need_ptr_mask,mask)

        # last operator is ptr, next should not still ptr
        mask[:, ptrs_begin:ptrs_begin + ptrs_num] = torch.where(last_is_prts, 0.,
                                                                   mask[:, ptrs_begin:ptrs_begin + ptrs_num])
        # if need parameter or cant be a full alg(choose,update,search), mask ptr
        need_para = (need_para_num != 0) | (operator_num<3)
        mask[:, ptrs_begin:ptrs_begin + ptrs_num] = torch.where(need_para, 0.,
                                                                mask[:, ptrs_begin:ptrs_begin + ptrs_num])
        # force choose end
        # make sure last operator is ptrs,and last normal operator is not crossover, then can choose 'end'
        need_end = (operator_num >=8) * (need_para_num == 0) * last_is_prts * (~last_normal_is_cross)
        need_end_mask = torch.zeros([dim0,total_index],dtype=torch.double).to(self.device)
        need_end_mask[:,end_index] = 1
        mask[:,:] = torch.where(need_end,need_end_mask,mask)

        # mask pso
        if discrete ==False:
            mask[:,5] = 0.

        # mask global search operator
        need_mask_opera = (have_gs == 1)*(need_para_num == 0)
        mask[:,only_gs_opera_index] = torch.where(need_mask_opera,0.,mask[:,only_gs_opera_index])
        # mask global search parameter
        need_mask_para = (have_gs == 1)*(need_para_num != 0)
        dim0,dim1 = need_mask_para.shape
        for i in range(dim0):
            if ~need_mask_para[i,0]: continue
            if discrete==False:
                if last_opera_index[i] == 6 or last_opera_index[i] == 7 or last_opera_index[i] == 8:
                    mask[i,31:] = 0
                elif last_opera_index[i] == 13 or last_opera_index[i] == 17:
                    mask[i, 31:] = 0
                elif last_opera_index[i] == 16:
                    if need_para_num[i,0] == 2:
                        mask[i, 31:] = 0
                    elif need_para_num[i,0] == 1:
                        mask[i, 28:31] = 0
            else:
                if last_opera_index[i] == 6 or last_opera_index[i] == 7 or last_opera_index[i] == 9:
                    mask[i, 21:27] = 0

        if ~torch.sum(mask, dim=1).all():
            print("have all zeros row!")
        return mask

    def judge_GS(self, have_gs, need_para_num, last_opera_index, trg):
        # judge alg have Global Search or not
        need_judge = (have_gs == 0) * (need_para_num == 0) * (last_opera_index >= search_begin) * (
                    last_opera_index <= search_begin + search_num - 1)
        dim0,dim1 = need_judge.shape
        for i in range(dim0):
            if ~need_judge[i,0]:continue
            last_opera_index_i = last_opera_index[i, 0]
            if only_gs_opera_index.__contains__(last_opera_index_i):
                have_gs[i, 0] = 1
            else:
                para_num = operator_para[voc_list[last_opera_index_i]]

                if discrete==False:
                    if para_num == 1:
                        para_index = trg[i, -1: ]
                        have_gs[i, 0] = ~is_ls_para(last_opera_index_i,para_index) + 0
                    elif para_num == 2:
                        para_index_1 = trg[i, -2:-1]
                        para_index_2 = trg[i, -1:]
                        one = ~is_ls_para(last_opera_index_i, para_index_1) + 0
                        two = ~is_ls_para(last_opera_index_i, para_index_2,2) + 0
                        have_gs[i, 0] = one|two + 0
                else:
                    if para_num == 1:
                        para_index = trg[i, -1: ]
                        have_gs[i, 0] = ~dis_is_ls_opera(last_opera_index_i,para_index) + 0

        return  have_gs

    def choose_unfinished(self,finished,have_gs,last_opera_index,last_normal_opera_index,operator_num,need_para_num,trg,action_emb,enc_src):
        unfinished_index = torch.where(finished == 0)[0]
        last_opera_index = torch.index_select(last_opera_index, 0, unfinished_index)
        last_normal_opera_index = torch.index_select(last_normal_opera_index, 0, unfinished_index)
        operator_num = torch.index_select(operator_num, 0, unfinished_index)
        need_para_num = torch.index_select(need_para_num, 0, unfinished_index)
        unfinished_trg = torch.index_select(trg, 0, unfinished_index)
        unfinished_have_gs = torch.index_select(have_gs, 0, unfinished_index)
        unfinished_action_emb = torch.index_select(action_emb, 0, unfinished_index)
        unfinished_enc_src = torch.index_select(enc_src, 0, unfinished_index)
        return unfinished_index,last_opera_index,last_normal_opera_index,operator_num,need_para_num,unfinished_trg,unfinished_have_gs,unfinished_action_emb,unfinished_enc_src

    def update_cur_alg_info(self,finished,unfinished_index,last_opera_index,last_normal_opera_index,operator_num,need_para_num,p,log_p,action_index,action_p,action_log_p,trg):
        # update info of cur alg
        is_opera = ((action_index >= operators_begin) * (action_index <= operators_begin + operators_num - 1)) | ((action_index >= ptrs_begin) * (action_index <= ptrs_begin + ptrs_num - 1))
        last_opera_index = torch.tensor(where(is_opera.cpu(), action_index.cpu(), last_opera_index.cpu())).to(
            self.device)
        # operator without ptrs
        is_normal_opera = ((action_index >= operators_begin) * (action_index <= operators_begin + operators_num - 1))
        last_normal_opera_index = torch.tensor(where(is_normal_opera.cpu(), action_index.cpu(), last_normal_opera_index.cpu())).to(
            self.device)
        operator_num = torch.tensor(where(is_opera.cpu(), operator_num.cpu() + 1, operator_num.cpu())).to(self.device)
        dim0, dim1 = is_opera.shape
        for i in range(dim0):
            if is_opera[i, 0]:
                need_para_num[i, 0] = operator_para[voc_list[action_index[i, 0]]]

        is_para = (action_index >= para_begin) * (action_index <= para_begin + para_num - 1)
        need_para_num = torch.tensor(where(is_para.cpu(), need_para_num.cpu() - 1, need_para_num.cpu())).to(
            self.device)


        # cat finished and unfinished rows to keep matrix
        cat_last_opera_index = torch.zeros(finished.shape, dtype=torch.int64).to(self.device)
        cat_last_opera_index.scatter_(0, torch.unsqueeze(unfinished_index, 1), last_opera_index)
        cat_last_normal_opera_index = torch.zeros(finished.shape, dtype=torch.int64).to(self.device)
        cat_last_normal_opera_index.scatter_(0, torch.unsqueeze(unfinished_index, 1), last_normal_opera_index)
        cat_operator_num = torch.zeros(finished.shape, dtype=torch.int64).to(self.device)
        cat_operator_num.scatter_(0, torch.unsqueeze(unfinished_index, 1), operator_num)
        cat_need_para_num = torch.zeros(finished.shape, dtype=torch.int64).to(self.device)
        cat_need_para_num.scatter_(0, torch.unsqueeze(unfinished_index, 1), need_para_num)
        last_opera_index = cat_last_opera_index
        last_normal_opera_index = cat_last_normal_opera_index
        operator_num = cat_operator_num
        need_para_num = cat_need_para_num

        # cat finished and unfinished rows to keep matrix
        cat_p = torch.ones(finished.shape).to(self.device)
        cat_p.scatter_(0, torch.unsqueeze(unfinished_index, 1), p)
        cat_log_p = torch.zeros(finished.shape).to(self.device)
        cat_log_p.scatter_(0, torch.unsqueeze(unfinished_index, 1), log_p)
        cat_action_index = torch.zeros(finished.shape, dtype=torch.int64).to(self.device)
        cat_action_index[:, :] = end_index
        cat_action_index.scatter_(0, torch.unsqueeze(unfinished_index, 1), action_index)

        action_p.append(cat_p)
        action_log_p.append(cat_log_p)
        trg = torch.cat([trg, cat_action_index], dim=1)

        finished = torch.tensor(where(cat_action_index.cpu() == end_index, 1, finished.cpu())).to(self.device)

        return last_opera_index,last_normal_opera_index,operator_num,need_para_num,action_p,action_log_p,trg,finished


