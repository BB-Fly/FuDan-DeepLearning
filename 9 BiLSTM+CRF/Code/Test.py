import DataLoad
import BiLSTM_CRF
import pickle



def Train():
    tag2idx, word2idx, train_set = DataLoad.load_train_dict_multi("../NER/English/train.txt","../NER/English/validation.txt")

    config = BiLSTM_CRF.Config(tag2idx, word2idx)
    config.num_epochs = 10


    model = BiLSTM_CRF.BiLSTM_CRF(config)

    model.fit(train_set)

    with open("ModelEng.pkl",'wb') as fp:
        pickle.dump(model,fp)




def Test(f_path):

    with open("ModelChi.pkl","rb") as fp1:
        model:BiLSTM_CRF.BiLSTM_CRF = pickle.load(fp1)
        test_list = DataLoad.load_test_list(model.model.tag2idx, model.model.word2idx ,f_path)
        idx2word = model.model.idx2word
        out = model.predict(test_list)

        with open("out.txt",'w',encoding='utf-8') as fp2:
            for i in range(len(test_list)):
                for j in range(len(test_list[i])):
                    s = idx2word.get(test_list[i][j],'__NULL__')+' '+out[i][j]+'\n'
                    fp2.write(s)
                fp2.write("\n")


if __name__ == "__main__":
    # Train()
    Test("../NER/chinese_test.txt")