import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import Dropout, Embedding
import torch.nn.functional as F
from torch.autograd import Variable
from load_data import *

class CNN(nn.Module):
    
    def __init__(self,char_size,char_dim,filter_size,char_out_dimension):
        super(CNN,self).__init__()
        self.char_size = char_size  
        self.char_dim = char_dim
        self.filter_size = filter_size
        self.out_channels = char_out_dimension
        self.char_embeds = nn.Embedding(self.char_size, self.char_dim)
        self.char_cnn = nn.Conv2d(in_channels= 1, out_channels=self.out_channels, kernel_size=(self.filter_size, self.char_dim))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(DROPOUT)
        self._init_weight()

    def _init_weight(self):
        init.xavier_uniform_(self.char_embeds.weight.data)
        
    def forward(self,inputs):
        """
        Arguments:
            inputs: [batch_size, word_len, char_len] 
        """
        bsz, word_len, char_len = inputs.size()
        inputs = inputs.type(torch.LongTensor)
        inputs  = inputs.view(-1, char_len)
        x = self.char_embeds(inputs)
        x = x.unsqueeze(1)
        x = self.char_cnn(x)
        x = self.relu(x)
        x = F.max_pool2d(x,kernel_size=(x.size(2), 1))
        x = self.dropout(x.squeeze())
        return x.view(bsz, word_len, -1)
    
class BILSTM(nn.Module):
    
    def __init__(self,word_size,word_dim,char_out_dimension,hidden_dim,target_size):
        super(BILSTM,self).__init__()
        self.word_size = word_size
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.out_channels = char_out_dimension
        self.target_size = target_size
        self.word_embeds = nn.Embedding(self.word_size, self.word_dim)
        self.lstm = nn.LSTM(self.word_dim+self.out_channels, self.hidden_dim, bidirectional=True,
                            batch_first = True,dropout=DROPOUT)
        self.linear = nn.Linear(self.hidden_dim*2,self.target_size)
        self._init_weights()

    def _init_weights(self, scope=1.):
        #self.word_embeds.weight.data.uniform_(-scope, scope)
        if PRE_TRAINED_EMBEDDING:
            self.word_embeds.weight.data.copy_(torch.from_numpy(pretrained_word_embeds))
            if NON_TRAINABLE:
                self.word_embeds.weight.requires_grad = False
            else:
                self.word_embeds.weight.requires_grad = True
        else:
            init.xavier_uniform_(self.word_embeds.weight.data)
        
    def forward(self, inputs,char_feats):
        """
        Arguments:
            inputs: [batch_size, seq_len] 
        """
        inputs = inputs.type(torch.LongTensor)
        x = self.word_embeds(inputs)
        x = torch.cat((char_feats, x), dim=-1)
        output,hidden = self.lstm(x)
        output = self.linear(output)
        return output
    
def log_sum_exp(input, keepdim=False):
    assert input.dim() == 2
    max_scores, _ = input.max(dim=-1, keepdim=True)
    output = input - max_scores
    return max_scores + torch.log(torch.sum(torch.exp(output), dim=-1, keepdim=keepdim))


def gather_index(input, index):
    assert input.dim() == 2 and index.dim() == 1
    index = index.unsqueeze(1).expand_as(input)
    output = torch.gather(input, 1, index)
    return output[:, 0]


class CRF(nn.Module):
    def __init__(self, label_size):
        super(CRF,self).__init__()
        self.label_size = label_size
        #self.torch = torch
        self.transitions = nn.Parameter(
            torch.randn(label_size, label_size))
        self._init_weight()

    def _init_weight(self):
        init.xavier_uniform_(self.transitions)
        self.transitions.data[label_mapping['<start>'], :].fill_(-10000.)
        self.transitions.data[:, label_mapping['<stop>']].fill_(-10000.)

    def _score_sentence(self, input, tags):
        bsz, sent_len, l_size = input.size()
        score = Variable(torch.FloatTensor(bsz).fill_(0.))
        s_score = Variable(torch.LongTensor([[label_mapping['<start>']]] * bsz))

        tags = torch.cat([s_score, tags], dim=-1)
        input_t = input.transpose(0, 1)

        for i, words in enumerate(input_t):
            temp = self.transitions.index_select(1, tags[:, i])
            bsz_t = gather_index(temp.transpose(0, 1), tags[:, i + 1])
            w_step_score = gather_index(words, tags[:, i + 1])
            score = score + bsz_t + w_step_score

        temp = self.transitions.index_select(1, tags[:, -1])
        bsz_t = gather_index(temp.transpose(0, 1),
                             Variable(torch.LongTensor([label_mapping['<stop>']] * bsz)))
        return score + bsz_t

    def forward(self, input):
        bsz, sent_len, l_size = input.size()
        init_alphas = torch.FloatTensor(
            bsz, self.label_size).fill_(-10000.)
        init_alphas[:, label_mapping['<start>']].fill_(0.)
        forward_var = Variable(init_alphas)

        input_t = input.transpose(0, 1)
        for words in input_t:
            alphas_t = []
            for next_tag in range(self.label_size):
                emit_score = words[:, next_tag].view(-1, 1)
                trans_score = self.transitions[next_tag].view(1, -1)
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var, True))
            forward_var = torch.cat(alphas_t, dim=-1)
        forward_var = forward_var + self.transitions[label_mapping['<stop>']].view(
            1, -1)
        return log_sum_exp(forward_var)

    def viterbi_decode(self, input):
        backpointers = []
        bsz, sent_len, l_size = input.size()

        init_vvars = torch.FloatTensor(
            bsz, self.label_size).fill_(-10000.)
        init_vvars[:, label_mapping['<start>']].fill_(0.)
        forward_var = Variable(init_vvars)

        input_t = input.transpose(0, 1)
        for words in input_t:
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(self.label_size):
                _trans = self.transitions[next_tag].view(
                    1, -1).expand_as(words)
                next_tag_var = forward_var + _trans
                best_tag_scores, best_tag_ids = torch.max(
                    next_tag_var, 1, keepdim=True)  # bsz
                bptrs_t.append(best_tag_ids)
                viterbivars_t.append(best_tag_scores)

            forward_var = torch.cat(viterbivars_t, -1) + words
            backpointers.append(torch.cat(bptrs_t, dim=-1))

        terminal_var = forward_var + self.transitions[label_mapping['<stop>']].view(1, -1)
        _, best_tag_ids = torch.max(terminal_var, 1)

        best_path = [best_tag_ids.view(-1, 1)]
        for bptrs_t in reversed(backpointers):
            best_tag_ids = gather_index(bptrs_t, best_tag_ids)
            best_path.append(best_tag_ids.contiguous().view(-1, 1))

        best_path.pop()
        best_path.reverse()

        return torch.cat(best_path, dim=-1)
    
class Model(nn.Module):
    
    def __init__(self,char_size, char_dim, filter_size, char_out_dimension,word_size, word_dim,
                 hidden_dim, target_size):
        super(Model,self).__init__()
        self.char_size = char_size
        self.char_dim = char_dim
        self.filter_size = filter_size
        self.char_out_dimension = char_out_dimension
        self.word_size = word_size
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.target_size = target_size
        self.cnn = CNN(self.char_size, self.char_dim, self.filter_size, self.char_out_dimension)
        self.bilstm = BILSTM(self.word_size, self.word_dim, self.char_out_dimension, self.hidden_dim, self.target_size)
        self.crf = CRF(self.target_size)

    def forward(self, words, chars, labels):
        char_feats = self.cnn(chars)
        output = self.bilstm(words, char_feats)
        pre_score = self.crf(output)
        label_score = self.crf._score_sentence(output, labels)
        return (pre_score - label_score).mean(), None

    def predict(self, word, char):
        char_out = self.cnn(char)
        out = self.bilstm(word, char_out)
        #out = self.logistic(lstm_out)
        return self.crf.viterbi_decode(out)
    
