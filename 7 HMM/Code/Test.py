import DataLoad
import HMM
import pickle


def Train():
    hmm = HMM.HMM()

    tag2idx, word2idx, train_set = DataLoad.load_train_dict_multi("../NER/English/train.txt", "../NER/English/validation.txt")
    

    hmm.pre_train(tag2idx,word2idx)
    hmm.fit(train_set)

    with open("ModelEng.pkl",'wb') as fp:
        pickle.dump(hmm,fp)



def Test():
    test_list = DataLoad.load_test_list("../NER/english_test.txt")
    with open("ModelEng.pkl","rb") as fp:
        hmm:HMM.HMM = pickle.load(fp)
        out = hmm.predict(test_list)

        with open("out.txt",'w',encoding='utf-8') as fp:
            for i in range(len(test_list)):
                for j in range(len(test_list[i])):
                    s = test_list[i][j]+' '+out[i][j]+'\n'
                    fp.write(s)
                fp.write("\n")



if __name__ =="__main__":

    # Train()
    Test()