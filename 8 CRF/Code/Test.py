import DataLoad
import CRF
import pickle



def Train():
    tag2idx, word2idx, train_set = DataLoad.load_train_dict_multi("../NER/English/train.txt","../NER/English/validation.txt")
    template = DataLoad.readTemplate("../NER/template_for_crf.utf8")
    crf = CRF.CRF()

    crf.pre_train(tag2idx,word2idx,template)
    crf.train(train_set,2)

    with open("ModelEng.pkl",'wb+') as fp:
        pickle.dump(crf,fp)




def Test():
    test_list = DataLoad.load_test_list("../NER/english_test.txt")
    with open("ModelEng.pkl","rb+") as fp:
        crf:CRF.CRF = pickle.load(fp)
        out = crf.predict(test_list)

        with open("out.txt",'w+',encoding='utf-8') as fp:
            for i in range(len(test_list)):
                for j in range(len(test_list[i])):
                    s = test_list[i][j]+' '+out[i][j]+'\n'
                    fp.write(s)
                fp.write("\n")


if __name__ == "__main__":
    #Train()
    Test()