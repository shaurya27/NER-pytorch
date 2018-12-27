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
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
#optimizer = ScheduledOptim(torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
#                                            weight_decay=0.05),LEARNING_RATE,len(train_dataloader.dataset.data))
def evaluate():
    model.eval()
    corrects ,eval_loss,_size,fscore = 0,0,0,0

    for word, char, label,_,_,word_max_len in tqdm.tqdm(valid_dataloader, mininterval=0.2,
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


def train():
    model.train()
    corrects ,total_loss,_size,fscore = 0,0,0,0
    for word, char, label,_,_,word_max_len in tqdm.tqdm(train_dataloader, mininterval=1,
                                  desc='Train Processing', leave=False):
        label = label.type(torch.LongTensor)
        word = Variable(word)
        char = Variable(char)
        label = Variable(label)
        optimizer.zero_grad()
        loss, _ = model(word, char, label)
        loss.backward()
        optimizer.step()
        pred = model.predict(word, char)
        corrects += (pred.data == label.data).sum()    
        total_loss += loss.data
        _size += train_dataloader.batch_size * word_max_len
        pre,rec,f_score,_ = precision_recall_fscore_support(label.data.numpy().reshape(-1,1),pred.data.numpy().reshape(-1,1),
                                                            average='macro') 
        fscore += f_score
        
    num_batches = (len(train_dataloader.dataset.data) // train_dataloader.batch_size)
    fscore = fscore/num_batches
    return total_loss / num_batches, corrects ,(float(corrects) / _size) * 100, _size,fscore

def save(filename):
    state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'valid_loss': train_loss,"word_dict": feature_mapping,
                "label_dict": label_mapping,"char_dict": character_mapping}
    torch.save(state, filename)

###########################################
###########################################

training_loss = []
validation_loss = []
train_accuracy = []
valid_accuracy =[]
best_fscore = None
total_start_time = time.time()

try:
    print('-' * 90)
    for epoch in range(1, NUM_EPOCH + 1):
        epoch_start_time = time.time()
        train_loss, train_corrects, train_acc, train_size,train_fscore = train()
        scheduler.step()
        training_loss.append(train_loss * 1000.)
        train_accuracy.append(train_acc/100.)

        print('| start of epoch {:3d} | time: {:2.2f}s | loss {:5.6f} | accuracy {:.4f}%({}/{}) | f1-score {:.4f}'.format(
            epoch, time.time() - epoch_start_time, train_loss, train_acc, train_corrects, train_size,train_fscore))

        valid_loss, valid_corrects, valid_acc, valid_size,valid_fscore = evaluate()

        validation_loss.append(valid_loss * 1000.)
        valid_accuracy.append(valid_acc / 100.)

        epoch_start_time = time.time()
        print('-' * 90)
        print('| end of epoch {:3d} | time: {:2.2f}s | loss {:.4f} | accuracy {:.4f}%({}/{} | f1-score {:.4f})'.format(
            epoch, time.time() - epoch_start_time, valid_loss, valid_acc, valid_corrects, valid_size,valid_fscore))
        print('-' * 90)
        if not best_fscore or best_fscore < valid_fscore:
            best_fscore = valid_fscore
            save('../save/checkpoint_epoch_'+str(epoch)+'_valid_loss_'+str(valid_loss)
              +'_valid_acc_'+str(valid_acc)+'_valid_fscore_'+str(valid_fscore)+'_'+'.pth.tar')
except KeyboardInterrupt:
    print("-" * 90)
    print("Exiting from training early | cost time: {:5.2f}min".format(
        (time.time() - total_start_time) / 60.0))
