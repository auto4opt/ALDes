import re
from collections import defaultdict
import os
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import copy

def read_data(filename):
    pattern = re.compile(r'\d+\.\d+|\d+')
    keyword1 = 'total env-steps'
    keyword2 = 'return mean'
    total_step =[]
    mean_var =[]
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if keyword1 in line:
                numbers = pattern.findall(line)
                total_step.append(int(numbers[-1]))
            if keyword2 in line:
                numbers = pattern.findall(line)
                numbers = [float(num) for num in numbers]
                mean_var.append(numbers)

    return_means = [d[-2] for d in mean_var]
    return_stds = [d[-1] for d in mean_var]
    return total_step,return_means,return_stds


def read_data_for_transformer(filename):
    pattern =  re.compile(r'-?\d+\.\d+|-?\d+')
    keyword = 'step :'
    total_step =[]
    data =[]
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            if keyword in line:
                line = line.split('step :')[1].split('Training')[0]
                numbers = pattern.findall(line)
                numbers = [float(num) for num in numbers]
                data.append(numbers)

    return data

def plot_each(plt,steps,means,stds,label,show_stds = False):
    plt.plot(steps, means, label=label,linewidth=2)
    if show_stds:
        plt.fill_between(steps,
                         [m - s for m, s in zip(means, stds)],
                         [m + s for m, s in zip(means, stds)],
                          alpha=0.2)
        
def read_from_pkl(pkls):
    datas = []
    for pkl in pkls:
        with open(pkl, 'rb') as f:
            loaded_data = pickle.load(f)
            datas.append(loaded_data)
    print(len(datas))
    return datas

def roll_and_draw(df,problem_set,save_path):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    _new_df = copy.deepcopy(df)
    new_df = _new_df.rolling(window=5, center=False).mean() 
    show_std = True
    for problem in problem_set:
        plt.figure(figsize=(5, 3))
        plot_each(plt, new_df['Epoch'], new_df[f'Means_{problem}'], new_df[f'Vars_{problem}'], f'problem{problem}',show_stds=show_std)
        plt.title(f'F{problem}')
        plt.xlabel('Episode')
        plt.ylabel('Performance')
        if show_std:
            plt.savefig(f'{save_path}\F{problem}.svg',dpi=300,format="svg",bbox_inches = 'tight')
        else:
            plt.savefig(f'{save_path}\F{problem}.svg',dpi=300,format="svg",bbox_inches = 'tight')

def roll_and_draw_mutil(dfs,problem_set,display_name,save_path):
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    new_dfs = []
    for df in dfs:
        _new_df = copy.deepcopy(df)
        new_df = _new_df.rolling(window=5, center=False).mean() 
        new_dfs.append(new_df)
        
    show_std = True
    for problem in problem_set:
        plt.figure(figsize=(5, 3))
        for new_df,_display_name in zip(new_dfs,display_name):
            plot_each(plt, new_df['Epoch'], new_df[f'Means_{problem}'], new_df[f'Vars_{problem}'], _display_name, show_stds=show_std)
        plt.title(f'F{problem}')
        plt.xlabel('Episode')
        plt.ylabel('Performance')
        plt.legend()
        path = f'{save_path}F{problem}.svg'
        print(path)
        if show_std:
            plt.savefig(f'{save_path}F{problem}.svg',dpi=300,format="svg",bbox_inches = 'tight')
        else:
            plt.savefig(f'{save_path}F{problem}.svg',dpi=300,format="svg",bbox_inches = 'tight')