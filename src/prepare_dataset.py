#import os
#sys.path.append('../')
from constant import * 

def read_corpus(lines):
    """
    convert corpus into features and labels
    """
    features = list()
    labels = list()
    tmp_fl = list()
    tmp_ll = list()
    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            line  = line.lower()
            line = line.rstrip('\n').split()
            tmp_fl.append(line[0])
            tmp_ll.append(line[-1])
        elif len(tmp_fl) > 0:
            features.append(tmp_fl)
            labels.append(tmp_ll)
            tmp_fl = list()
            tmp_ll = list()
    if len(tmp_fl) > 0:
        features.append(tmp_fl)
        labels.append(tmp_ll)

    return features, labels

#if __name__ == "__main__":
    
# Training data
with open(training_data_path,'r') as f:
    train_lines = f.readlines()
train_feature, train_labels = read_corpus(train_lines)
train_data = zip(train_feature,train_labels)
# Validation data
with open(validation_data_path,'r') as f:
    valid_lines = f.readlines()
valid_feature, valid_labels = read_corpus(valid_lines)
valid_data = zip(valid_feature,valid_labels)
# Test data
with open(test_data_path,'r') as f:
    test_lines = f.readlines()
test_feature, test_labels = read_corpus(test_lines)
test_data = zip(test_feature,test_labels)
    