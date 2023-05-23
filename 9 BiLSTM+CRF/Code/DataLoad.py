import copy
from torch.utils.data import Dataset
import torch


def load_train_dict_multi(*f_paths)->tuple[dict,dict,list]:
    '''get tag2idx, word2idx, train_set:[word -> tag]'''
    tag_cnt=2
    word_cnt=3
    tag2idx = {'__BEGIN__':0, '__END__':1,}
    word2idx = {'__BEGIN__':0, '__END__':1, '__NULL__': 2}
    train_set = [[],[]]

    for f_path in f_paths:
        with open(f_path,'r',encoding='utf-8') as fp:
            lines = fp.readlines()
            tmp1:list = []
            tmp2:list = []
            for line in lines:
                s = line[:-1].split(' ')
                if len(s)<2:
                    train_set[0].append(copy.deepcopy(tmp1))
                    train_set[1].append(copy.deepcopy(tmp2))
                    tmp1.clear()
                    tmp2.clear()
                    continue     
                if word2idx.get(s[0])==None:
                    word2idx[s[0]] = word_cnt
                    word_cnt += 1
                if tag2idx.get(s[1])==None:
                    tag2idx[s[1]] = tag_cnt
                    tag_cnt += 1
                tmp1.append(word2idx[s[0]])
                tmp2.append(tag2idx[s[1]])
            if(len(tmp1)):
                train_set[0].append(copy.deepcopy(tmp1))
                train_set[1].append(copy.deepcopy(tmp2))

    return tag2idx, word2idx, train_set


def load_test_list(tag2idx, word2idx:dict, *f_paths):

    train_set = []

    for f_path in f_paths:
        with open(f_path,'r',encoding='utf-8') as fp:
            lines = fp.readlines()
            tmp1:list = []
            for line in lines:
                s = line[:-1].split(' ')
                if len(s)<2:
                    train_set.append(copy.deepcopy(tmp1))
                    tmp1.clear()
                    continue
                tmp1.append(word2idx.get(s[0],2))
            if(len(tmp1)):
                train_set.append(copy.deepcopy(tmp1))

    return train_set