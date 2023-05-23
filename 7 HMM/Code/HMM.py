import numpy as np



class HMM:
    def __init__(self) -> None:
        # Map
        self.tag2idx:dict = None
        self.str2idx:dict = None
        self.idx2tag:dict = None

        # Size
        self.tag_size:int = None
        self.str_size:int = None

        # Matrix
        self.transition:np.array = None
        self.emission:np.array = None
        self.pi:np.array = None

        # Const
        self.div_offt = 1e-8



    def pre_train(self, tags:dict, words:dict):
        '''use this func to get:\n map: tag->idx\n map: word->idx\n map: idx->tag'''

        self.tag2idx = tags
        self.str2idx = words
        self.idx2tag = {k:v for v, k in self.tag2idx.items()}

        self.tag_size = len(self.tag2idx)
        self.str_size = len(self.str2idx)

        self.transition = np.zeros((self.tag_size,self.tag_size))
        self.emission = np.zeros((self.tag_size,self.str_size))
        self.pi = np.zeros((self.tag_size))



    def fit(self, train_set:list[list[tuple[int,int]]]):

        self.__estimate_emission(train_set)
        self.__estimate_transition(train_set)
        self.__estimate_pi(train_set)

        self.pi = np.log(self.pi)
        self.transition = np.log(self.transition)
        self.emission = np.log(self.emission)
        

    
    def predict(self,text:list[list[str]])->list[list[str]]:
        res = []

        for txt in text:
            best_tag_id = self.__decode(txt)
            best_tag = []
            for tag_id in best_tag_id:
                best_tag.append(self.idx2tag[tag_id])
            res.append(best_tag)

        return res



    def __estimate_transition(self,train_set:list[list[tuple[int,int]]]):
        '''use this func to init'''
        for lst in train_set:
            _, cur_tag = lst[0]
            for i in range(1,len(lst)):
                _, nxt_tag = lst[i]
                self.transition[cur_tag][nxt_tag] += 1.
                cur_tag = nxt_tag

        self.transition[self.transition==0] = self.div_offt
        self.transition /= np.sum(self.transition,axis=1,keepdims=True)

        



    def __estimate_pi(self, train_set:list[list[tuple[int,int]]]):
        '''use this func to init'''
        for lst in train_set:
            _, cur_tag = lst[0]
            self.pi[cur_tag] += 1.

        self.pi[self.pi==0] = self.div_offt
        self.pi /= np.sum(self.pi)
            


    def __estimate_emission(self,train_set:list[list[tuple[int,int]]]):
        '''use this func to init'''
        for lst in train_set:
            for word2tag in lst:
                word, tag = word2tag
                self.emission[tag][word] += 1
        
        self.emission[self.emission==0] = self.div_offt
        self.emission /= np.sum(self.emission,axis=1, keepdims=True)



    def __get_pro(self, word:str):
        '''get the p(word|state)'''
        
        if self.str2idx.get(word)==None:
            res = np.log(np.ones(self.tag_size)/self.tag_size)
        else:
            res = np.ravel(self.emission[:,self.str2idx[word]])

        return res
    


    def __decode(self,text:list[str]):
        '''try decode with viterbi'''
        size = len(text)
        cur_table = np.zeros((size, self.tag_size))
        nxt_table = np.zeros((size, self.tag_size))

        cur_table[0,:] = self.pi + self.__get_pro(text[0])

        for i in range(1,size):
            p_state = self.__get_pro(text[i])
            p_state = np.expand_dims(p_state,axis=0)

            pre_score = np.expand_dims(cur_table[i-1, :], axis=-1)
            cur_score = pre_score + self.transition + p_state
            
            cur_table[i, :] = np.max(cur_score, axis=0)
            nxt_table[i, :] = np.argmax(cur_score, axis=0)

        best_tag = int(np.argmax(cur_table[-1, :]))
        best_tags = [best_tag, ]
        for i in range(size-1, 0, -1):
            best_tag = int(nxt_table[i, best_tag])
            best_tags.append(best_tag)
        return list(reversed(best_tags))



