import copy



def load_train_dict(f_path:str)->tuple[dict,dict,list[list[tuple[int,int]]]]:
    tag_cnt=0
    word_cnt=0
    tag2idx = {}
    word2idx = {}
    train_set = []


    with open(f_path,'r',encoding='utf-8') as fp:
        lines = fp.readlines()
        tmp:list = []
        for line in lines:
            s = line[:-1].split(' ')
            if len(s)<2:
                train_set.append(copy.deepcopy(tmp))
                tmp.clear()
                continue     
            if word2idx.get(s[0])==None:
                word2idx[s[0]] = word_cnt
                word_cnt += 1
            if tag2idx.get(s[1])==None:
                tag2idx[s[1]] = tag_cnt
                tag_cnt += 1
            tmp.append((word2idx[s[0]],tag2idx[s[1]]))
        if(len(tmp)):
            train_set.append(copy.deepcopy(tmp))

    return tag2idx, word2idx, train_set


def load_test_list(f_path:str)->list[list[str]]:

    res = []

    with open(f_path, 'r',encoding='utf-8') as fp:
        lines = fp.readlines()
        tmp:list = []
        for line in lines:
            s =  line[:-1].split(" ")
            if(len(s)>=2):
                tmp.append(s[0])
            else:
                res.append(copy.deepcopy(tmp))
                tmp.clear()
        if(len(tmp)):
            res.append(copy.deepcopy(tmp))

    return res



def load_train_dict_multi(*f_paths):
    tag_cnt=0
    word_cnt=0
    tag2idx = {}
    word2idx = {}
    train_set = []

    for f_path in f_paths:
        with open(f_path,'r',encoding='utf-8') as fp:
            lines = fp.readlines()
            tmp:list = []
            for line in lines:
                s = line[:-1].split(' ')
                if len(s)<2:
                    train_set.append(copy.deepcopy(tmp))
                    tmp.clear()
                    continue     
                if word2idx.get(s[0])==None:
                    word2idx[s[0]] = word_cnt
                    word_cnt += 1
                if tag2idx.get(s[1])==None:
                    tag2idx[s[1]] = tag_cnt
                    tag_cnt += 1
                tmp.append((word2idx[s[0]],tag2idx[s[1]]))
            if(len(tmp)):
                train_set.append(copy.deepcopy(tmp))

    return tag2idx, word2idx, train_set
