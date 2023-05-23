import copy



def load_train_dict(f_path:str)->tuple[dict,dict,list]:
    '''get tag2idx, word2idx, train_set:[word -> tag]'''
    tag_cnt=1
    word_cnt=0
    tag2idx = {'NULL':0,}
    word2idx = {}
    train_set = [[],[]]


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
            tmp1.append(s[0])
            tmp2.append(tag2idx[s[1]])
        if(len(tmp1)):
            train_set[0].append(copy.deepcopy(tmp1))
            train_set[1].append(copy.deepcopy(tmp2))

    return tag2idx, word2idx, train_set



def load_train_dict_multi(*f_paths)->tuple[dict,dict,list]:
    '''get tag2idx, word2idx, train_set:[word -> tag]'''
    tag_cnt=1
    word_cnt=0
    tag2idx = {'NULL':0,}
    word2idx = {}
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
                tmp1.append(s[0])
                tmp2.append(tag2idx[s[1]])
            if(len(tmp1)):
                train_set[0].append(copy.deepcopy(tmp1))
                train_set[1].append(copy.deepcopy(tmp2))

    return tag2idx, word2idx, train_set



def load_test_list(f_path:str)->list:
    '''get test list [[sentence]]'''

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


def get_str_btw(s, b, e):
    '''exp: func:\n\n %x[1,0]/x[0,0]\n\n-> [1,0]'''
    a = list()
    index1 = 0
    index2 = 0
    for x in range(len(s)):
        if s[x] == b:
            index1 = x
            continue
        if s[x] == e:
            index2 = x
            a.append(int(s[index1+1:index2]))
    return a
    

def readTemplate(fileName):
    '''get template list, exp: func:\n\n[%x[1,0]/x[0,0], ...]\n\n-> [[1,0], ...]'''
    gram = list()
    unigram=list()
    bigram=list()
    with open(fileName,'r',encoding='UTF-8') as fp:
        lines = fp.readlines()
        for line in lines:
            if len(line)>1:
                term = get_str_btw(line,"[",",")
                if (line[0] == "U")and(len(term)>0):
                    unigram.append(term)
                if (line[0] == "B")and(len(term)>0):
                    bigram.append(term)
    gram.append(unigram)
    gram.append(bigram)

    return gram