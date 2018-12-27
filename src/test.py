from model import *

import numpy as np
from torch.autograd import Variable
from sklearn.metrics import f1_score,precision_score,precision_recall_fscore_support
import time
import tqdm

CHAR_SIZE = len(character_mapping)
WORD_SIZE = len(feature_mapping)
TARGET_SIZE = len(label_mapping)

model = Model(CHAR_SIZE, CHAR_DIM, FILTER_SIZE, CHAR_OUT_DIMENSION, WORD_SIZE, WORD_DIM, HIDDEN_DIM, TARGET_SIZE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

def test():
    model.eval()
    corrects ,eval_loss,_size,fscore = 0,0,0,0

    for word, char, label,_,_,word_max_len in tqdm.tqdm(test_dataloader, mininterval=0.2,
                                  desc='Evaluate Processing', leave=False):
        label = label.type(torch.LongTensor)
        word = Variable(word)
        char = Variable(char)
        label = Variable(label)
        loss, _ = model(word, char, label)
        pred = model.predict(word, char)

        eval_loss += loss.data.item()

        corrects += (pred.data == label.data).sum()
        _size += valid_dataloader.batch_size * word_max_len
        pre,rec,f_score,_ = precision_recall_fscore_support(label.data.numpy().reshape(-1,1),pred.data.numpy().reshape(-1,1),
                                                            average='macro') 
        fscore += f_score

    num_batches = (len(valid_dataloader.dataset.data) // valid_dataloader.batch_size)
    fscore = fscore/num_batches
    return eval_loss / num_batches, corrects,(float(corrects) / _size) * 100, _size, fscore

checkpoint = torch.load(MODEL_PATH,map_location=lambda storage, loc: storage)
model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])

try:
    print('-' * 90)
    epoch_start_time = time.time()

    valid_loss, valid_corrects, valid_acc, valid_size,valid_fscore = test()

    epoch_start_time = time.time()
    print('-' * 90)
    print('time: {:2.2f}s | loss {:.4f} | accuracy {:.4f}%({}/{} | f1-score {:.4f})'.format(time.time() - epoch_start_time, valid_loss, valid_acc, valid_corrects, valid_size,valid_fscore))
    print('-' * 90)
except KeyboardInterrupt:
    print("-" * 90)
    print("Exiting from training early | cost time: {:5.2f}min".format(
        (time.time() - total_start_time) / 60.0))


