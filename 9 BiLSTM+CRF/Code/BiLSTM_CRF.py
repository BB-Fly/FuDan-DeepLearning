import torch
import torch.nn as nn
import numpy as np
import copy


class Config:
    def __init__(self, tag2idx:dict, word2idx:dict):
        # dict
        self.tag2idx = tag2idx
        self.word2idx = word2idx
        self.idx2tag = {i: label for label, i in self.tag2idx.items()}
        self.idx2word = {i:word for word, i in self.word2idx.items()}

        # len
        self.num_tag = len(tag2idx)
        self.num_word = len(word2idx)

        # train and eval
        self.learning_rate = 0.01
        self.num_epochs = 10
        self.weight_decay = 0.0001




class _bilstm_crf(nn.Module):

    def __init__(self, config:Config):
        super(_bilstm_crf, self).__init__()

        # dict
        self.idx2tag = config.idx2tag
        self.idx2word = config.idx2word
        self.tag2idx = config.tag2idx
        self.word2idx = config.word2idx

        # num
        self.num_tag = config.num_tag
        self.num_word = config.num_word

        # begin, end
        self.start_tag_id = config.tag2idx["__BEGIN__"]
        self.end_tag_id = config.tag2idx["__END__"]

        # learning pra
        self.learning_rate = config.learning_rate
        self.num_epochs = config.num_epochs
        self.weight_decay = config.weight_decay

        # embedding + BiLSTM + Linear
        self.embedding = nn.Embedding(self.num_word, 500)
        self.encoder = nn.LSTM(500, 200, num_layers=1, bidirectional=True, batch_first =True)
        self.linear = nn.Linear(2*200, self.num_tag)
        

        # t[i][j] means state j to i
        self.transitions = nn.Parameter(torch.randn(self.num_tag, self.num_tag))
        
        self.transitions.data[self.start_tag_id, :] = -10000.
        self.transitions.data[:, self.end_tag_id] = -10000.
        self.hidden = torch.randn(2, 1, 200), torch.randn(2, 1, 200)



    def _get_lstm_features(self, input_ids:torch.Tensor):
        # inputs -> feats
        embeds = self.embedding(input_ids).view(1, input_ids.shape[1], -1)
        self.encoder.flatten_parameters()
        self.hidden = torch.randn(2, 1, 200), torch.randn(2, 1, 200)
        encoder_out, _ = self.encoder(embeds, self.hidden)
        decoder_out = encoder_out.view(input_ids.shape[1], -1)
        lstm_logits = self.linear(decoder_out)
        return lstm_logits

    def log_sum_exp(self, smat:torch.Tensor):
        vmax = smat.max(dim=0, keepdim=True).values
        
        return torch.log(torch.sum(torch.exp(smat - vmax), axis=0, keepdim=True)) + vmax

    def _forward_alg(self, feats):

        alphas = torch.full((1, self.num_tag), -10000.)
        alphas[0][self.start_tag_id] = 0.


        for feat in feats:
            alphas = self.log_sum_exp(alphas.T + self.transitions + feat.unsqueeze(0))

        score = self.log_sum_exp(alphas.T + 0 + self.transitions[:, self.end_tag_id].view(-1, 1))
        return score.flatten()

    def _score_sentence(self, feats, tags):

        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.start_tag_id], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score += self.transitions[tags[i], tags[i+1]] + feat[tags[i + 1]]
        
        score += self.transitions[tags[-1], self.end_tag_id]
        return score
    

    def _viterbi_decode(self, feats):
        backtrace = []
        alpha = torch.full((1, self.num_tag), -10000.)
        alpha[0][self.start_tag_id] = 0

        for frame in feats:
            smat = alpha.T + frame.unsqueeze(0) + self.transitions
            backtrace.append(smat.argmax(0))
            alpha = smat.max(dim=0, keepdim=True).values
        # Transition to STOP_TAG
        smat = alpha.T + 0 + self.transitions[:, self.end_tag_id].view(-1, 1)
        best_tag_id = smat.flatten().argmax().item()
        best_score = smat.max(dim=0, keepdim=True).values.item()
        best_path = [best_tag_id]

        for bptrs_t in reversed(backtrace[1:]): 
            best_tag_id = bptrs_t[best_tag_id].item()
            best_path.append(best_tag_id)
        best_path.reverse()
        return best_score, best_path

    def forward(self, sentence_ids, tags_ids):
        tags_ids = tags_ids.view(-1)
        feats = self._get_lstm_features(sentence_ids)
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags_ids)
        outputs = (forward_score - gold_score, )
        _, tag_seq = self._viterbi_decode(feats)
        outputs = (tag_seq, ) + outputs
        return outputs

    def predict(self, sentence_ids):
        lstm_feats = self._get_lstm_features(sentence_ids)
        _, tag_seq = self._viterbi_decode(lstm_feats)
        return tag_seq



class BiLSTM_CRF:
    def __init__(self, config:Config):
        self.model = _bilstm_crf(config)


    def fit(self, train_set):

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.model.learning_rate, weight_decay=self.model.weight_decay)
        sentences = train_set[0]
        tags = train_set[1]

        for epoch in range(self.model.num_epochs):
            for i in range(len(tags)):
                sentence = torch.LongTensor(sentences[i]).reshape((1,len(sentences[i])))
                tag = torch.LongTensor(tags[i]).reshape((1,len(tags[i])))
                self.model.train()
                outputs_logits, loss = self.model(sentence, tag)
                self.model.zero_grad()
                loss.backward()
                optimizer.step()
            print(epoch)

        pass


    def predict(self, test_set):
        res = []

        with torch.no_grad():
            for sentence in test_set:
                sen = torch.LongTensor(sentence).reshape((1,len(sentence)))
                feats = self.model._get_lstm_features(sen)
                _, ans = self.model._viterbi_decode(feats)
                for i in range(len(ans)):
                    ans[i] = self.model.idx2tag[ans[i]]
                res.append(copy.deepcopy(ans))
        return res