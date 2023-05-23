import numpy as np



class CRF:
    def  __init__(self) -> None:

        # Map
        self.tag2idx:dict = None
        self.str2idx:dict = None
        self.idx2tag:dict = None

        self.ScoreMap:dict = {}

        # Size
        self.tag_size:int = None
        self.str_size:int = None

        # Template
        self.template:list[list[list]] = None

        # Const
        self.div_offt = 1e-8
        
        

    def pre_train(self, tags:dict, words:dict, template:list):
        '''use this func to get:\n map: tag->idx\n map: word->idx\n map: idx->tag'''

        self.tag2idx = tags
        self.str2idx = words
        self.idx2tag = {k:v for v, k in self.tag2idx.items()}

        self.tag_size = len(self.tag2idx)
        self.str_size = len(self.str2idx)

        self.template = template



    def train(self, data:list, iter=1):
        sentences = data[0]
        tags = data[1]
        for it in range(iter):
            wr = 0
            totalTest = 0
            for i in range(len(sentences)):
                sentence = sentences[i]
                totalTest+= len(sentence)
                tag = tags[i]
                if len(sentence)==0:
                    continue
                wr += self._train(sentence, tag)
                corrNum = totalTest - wr
                print("iter: {}, i: {}\taccuracy:{}".format(it,i,corrNum/totalTest))



    def predict(self, sentences:list):
        res = []
        for sentence in sentences:
            res.append(self._decode(sentence))
        
        return res
    
    
    
    def _makeKey(self, template, identity:str, sentence:list, pos:int, statusCovered:str):
        string = ""
        string +=  identity
        for offset in template:
            index = pos + offset
            if ((index < 0)or(index >= len(sentence))):
                string+=" "
            else:
                string+=sentence[index]
            string+="/"
        string+=str(statusCovered)
        return string
    

    def _makeBiKey(self,template, idx:int, sentence, pos, preS:int, thisS:int):
        return self._makeKey(template,str(idx),sentence,pos, str(preS)+'-'+str(thisS))
    

    def _makeUiKey(self,template, idx:int, sentence, pos, thisS:int):
        return self._makeKey(template,str(idx),sentence,pos, str(thisS))


    
    def _getBiScore(self, sentence:list, thisPos:int, preStatus:int, thisStatus:int):
        biScore = 0
        biTemplate = self.template[1]
        num = len(biTemplate)
        for i in range(num):
            key = self._makeBiKey(biTemplate[i], i, sentence, thisPos, preStatus , thisStatus)
            if(key in self.ScoreMap.keys()):
                biScore += self.ScoreMap[key]
        return biScore
    


    def _getUniScore(self, sentence:list, thisPos:int, thisStatus:int):
        uniScore = 0
        uniTemplate = self.template[0]
        num = len(uniTemplate)
        for i in range(num):
            key = self._makeUiKey(uniTemplate[i], i, sentence, thisPos, thisStatus)
            if (key in self.ScoreMap.keys()):
                uniScore += self.ScoreMap[key]
        return uniScore
    
    
    
    def _decode(self, sentence):
        '''viterbi decode'''
        lenth = len(sentence)
        statusFrom = [["" for col in range(lenth)] for row in range(self.tag_size)] #col: idx in sentence
        maxScore = [[-1 for col in range(lenth)] for row in range(self.tag_size)] # row: idx in tag
        for col in range(lenth):
            for row in range(self.tag_size):
                thisStatus = row
                if(col == 0):
                    uniScore = self._getUniScore(sentence, 0, thisStatus)
                    biScore = self._getBiScore(sentence, 0, 0, thisStatus)
                    maxScore[row][0] = uniScore + biScore
                else:
                    scores = []
                    for i in range(self.tag_size):
                        preStatus = i
                        transScore = maxScore[i][col - 1]
                        uniScore = self._getUniScore(sentence, col, thisStatus)
                        biScore = self._getBiScore(sentence, col, preStatus, thisStatus)
                        scores.append(transScore + uniScore + biScore)
                        maxIndex = np.argmax(scores)
                        maxScore[row][col] = scores[maxIndex]
                        statusFrom[row][col] = self.idx2tag[maxIndex]
        resBuf = ['' for col in range(lenth)]
        scoreBuf = [0 for col in range(self.tag_size)]
        for i in range(self.tag_size):
            scoreBuf[i] = maxScore[i][lenth - 1]
        resBuf[lenth - 1] = self.idx2tag[np.argmax(scoreBuf)]
        for backIndex in range(lenth-2, -1 , -1):
            resBuf[backIndex] = statusFrom[self.tag2idx[resBuf[backIndex + 1]]][backIndex + 1]
        temp = []
        for i in range(lenth):
            temp.append(resBuf[i])
        return temp
    
    def _train(self, sentence, theoryRes):
        
        myRes = self._decode(sentence)
        length = len(sentence)
        wrongNum = 0
        for i in range(length):
            myResI = self.tag2idx[myRes[i]]
            theoryResI = theoryRes[i]
            if (myResI!=theoryResI):
                wrongNum = wrongNum+1
                # update Unigram template
                uniTem = self.template[0]
                uniNum = len(uniTem)
                for uniIndex in range(uniNum):
                    uniMyKey = self._makeUiKey(uniTem[uniIndex], uniIndex, sentence, i, myResI)
                    if (not (uniMyKey in self.ScoreMap.keys())):
                        self.ScoreMap[uniMyKey] = -1
                    else:
                        myRawVal = self.ScoreMap[uniMyKey]
                        self.ScoreMap[uniMyKey] = myRawVal - 1
                    uniTheoryKey = self._makeUiKey(uniTem[uniIndex], uniIndex, sentence, i, theoryResI)
                    if (not (uniTheoryKey in self.ScoreMap.keys())):
                        self.ScoreMap[uniTheoryKey] = 1
                    else:
                        theoryRawVal = self.ScoreMap[uniTheoryKey]
                        self.ScoreMap[uniMyKey] = theoryRawVal + 1
                # update Bigram template
                biTem = self.template[1]
                biNum = len(biTem)
                for biIndex in range(biNum):
                    biMyKey = ""
                    biTheoryKey = ""
                    if i >= 1:
                        biMyKey = self._makeBiKey(biTem[biIndex], biIndex, sentence, i, self.tag2idx[(myRes[i - 1])],self.tag2idx[(myRes[i])])
                        biTheoryKey = self._makeBiKey(biTem[biIndex], biIndex, sentence, i, theoryRes[i - 1],theoryRes[i])
                    else:
                        biMyKey = self._makeBiKey(biTem[biIndex], biIndex, sentence, i, 0,self.tag2idx[(myRes[i])])
                        biTheoryKey = self._makeBiKey(biTem[biIndex], biIndex, sentence, i, 0,theoryRes[i])
                    if (not (biMyKey in self.ScoreMap.keys())):
                        self.ScoreMap[biMyKey] = -1
                    else:
                        myRawVal = self.ScoreMap[biMyKey]
                        self.ScoreMap[biMyKey] = myRawVal - 1
                    if (not (biTheoryKey in self.ScoreMap.keys())):
                        self.ScoreMap[biTheoryKey] = 1
                    else:
                        theoryRawVal = self.ScoreMap[biTheoryKey]
                        self.ScoreMap[biTheoryKey] = theoryRawVal + 1
        
        return wrongNum
    
    