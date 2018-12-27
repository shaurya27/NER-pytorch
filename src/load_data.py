from prepare_dataset import *
from torch.utils.data import DataLoader
import numpy as np
import torch
import codecs

def create_mapping(item_list,min_count = 1):
    '''
    create mapping from list of list
    '''
    word_count = {}
    word_map = {}
    for items in item_list:
        for item in items:
            if word_count.get(item):
                word_count[item]+=1
            else:
                word_count[item] = 1
    for k,v in word_count.iteritems():
        if v>min_count:
            word_map[k] = len(word_map)+1
    word_map['<pad>'] = 0
    return word_map


## Create Dictionaries
feature_mapping = create_mapping(train_feature,min_count=1)
feature_mapping['<unk>'] = len(feature_mapping)
print "Length of words dictionary : {}".format(len(feature_mapping))

label_mapping = create_mapping(train_labels,min_count=1)
label_mapping['<start>'] = len(label_mapping)
label_mapping['<stop>'] = len(label_mapping)
print "Length of label dictionary : {}".format(len(label_mapping))

words = [word for word in feature_mapping]
character_mapping = create_mapping(words,min_count=1) 
character_mapping['<unk>'] = len(character_mapping)
print "Length of character dictionary : {}".format(len(character_mapping))

################################################
## pre trained embedding
#################################################
all_word_embeds = {}
for i, line in enumerate(codecs.open(PRE_TRAINED_EMBEDDING_PATH, 'r', 'utf-8')):
    s = line.strip().split()
    if len(s) == WORD_DIM + 1:
        all_word_embeds[s[0]] = np.array([float(i) for i in s[1:]])

#Intializing Word Embedding Matrix
pretrained_word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(feature_mapping), WORD_DIM))

for w in feature_mapping:
    if w.lower() in all_word_embeds:
        pretrained_word_embeds[feature_mapping[w]] = all_word_embeds[w.lower()]

print('Loaded %i pretrained embeddings.' % len(all_word_embeds))
## To save memory
del all_word_embeds






####################################
### Dataloader
####################################
class CustomDataset():
    
    def __init__(self,data,word2id,tag2id,char2id):
        self.data = data
        self.word2id = word2id
        self.tag2id = tag2id
        self.char2id = char2id
        
        
    def __getitem__(self,index):
        X_word = self.data[index][0]
        y_label = self.data[index][1]
        X = [self.word2id.get(word) if self.word2id.get(word) else self.word2id['<unk>'] for word in X_word]
        #X = torch.tensor(X)
        y = [self.tag2id.get(tag) if self.tag2id.get(tag) else self.tag2id['<unk>'] for tag in y_label]
        #y = torch.tensor(y)
        X_char = [[self.char2id[c] if self.char2id.get(c) else self.char2id['<unk>'] for c in word] if self.word2id.get(word) else [self.char2id['<unk>']] for word in X_word]
        chars2_length = [len(c) for c in X_char]
        char_maxl = max(chars2_length)
        word_len = len(X_word)
        return {'words': X_word,'labels':y_label,'word_id':X,'label_id':y,'char_id':X_char,"max_len": char_maxl,"word_len":word_len}
        
    def __len__(self):
        return len(self.data)
    
def collate_fn(batch):
    max_len = [item['max_len'] for item in batch]
    max_word = [item['word_len'] for item in batch]
    cahr_max_len = max(max_len)
    word_max_len = max(max_word)
    #data_word = [item['word_id'] for item in batch]
    char_data = np.zeros((len(batch),word_max_len,cahr_max_len))
    word_data = np.zeros((len(batch),word_max_len))
    target_data = np.zeros((len(batch),word_max_len))
    #char_word = [char_data[i,j,:] for i,item in enumerate(batch) for j,c in item['char_id'] ]
    for i,item in enumerate(batch):
        for j,c in enumerate(item['char_id']):
            char_data[i,j,:len(c)] = c
    for i,item in enumerate(batch):
        word_data[i,:len(item['word_id'])] = item['word_id']
        target_data[i,:len(item['label_id'])] = item['label_id']
    target = [item['label_id'] for item in batch]
    actual_word =[item['words'] for item in batch]
    actual_label = [item['labels'] for item in batch]
    return torch.tensor(word_data),torch.tensor(char_data),torch.tensor(target_data),actual_word,actual_label,word_max_len

### Load datas
train_dataloader = DataLoader(CustomDataset(train_data,feature_mapping,label_mapping,character_mapping),
                              batch_size=BATCH_SIZE,collate_fn = collate_fn,shuffle=True)
valid_dataloader = DataLoader(CustomDataset(valid_data,feature_mapping,label_mapping,character_mapping),
                              batch_size=BATCH_SIZE,collate_fn = collate_fn,shuffle=True)
test_dataloader = DataLoader(CustomDataset(test_data,feature_mapping,label_mapping,character_mapping),
                              batch_size=BATCH_SIZE,collate_fn = collate_fn,shuffle=True)