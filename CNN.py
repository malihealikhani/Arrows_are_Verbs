import codecs
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
use_cuda = torch.cuda.is_available()


all_data_train = pd.read_csv('./fold3/train3')
all_data_test = pd.read_csv('./fold3/test3')


data=[]
for l in all_data_train.itertuples():
    sent=l[3]
    sent = sent.rstrip()
    data.append((sent.split(' '), l[2]))


data_test=[]
for l in all_data_test.itertuples():
    sent=l[3]
    sent = sent.rstrip()
    data_test.append((sent.split(' '), l[2]))


max_sentence_len = max([len(data) for sentence, _ in data])
print('sentence maxlen', max_sentence_len)

vocab = []
for d, _ in data:
    for w in d:
        if w not in vocab: vocab.append(w)
vocab = sorted(vocab)
vocab_size = len(vocab)
w2i = {w:i for i,w in enumerate(vocab)}
i2w = {i:w for i,w in enumerate(vocab)}
div_idx = (int)(len(data) * 0.8)
random.shuffle(data)
train_data = data[:div_idx]
test_data = data[div_idx:]


class Net(nn.Module):
    def __init__(self, vocab_size, embd_size, out_chs, filter_heights):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embd_size)
        self.conv = nn.ModuleList([nn.Conv2d(1, out_chs, (fh, embd_size)) for fh in filter_heights])
        self.dropout = nn.Dropout(.5)
        self.fc1 = nn.Linear(out_chs*len(filter_heights), 1)
        
    def forward(self, x):
        x = self.embedding(x) 
        x = x.unsqueeze(1) 
        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv]
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1) 
        x = self.dropout(x)
        x = self.fc1(x)
        probs = F.sigmoid(x)
        return probs


def train(model, data, batch_size, n_epoch):
    model.train() 
    if use_cuda:
        model.cuda()
    losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    for epoch in range(n_epoch):
        epoch_loss = 0.0
        random.shuffle(data)
        for i in range(0, len(data)-batch_size, batch_size):
            in_data, labels = [], []
            for sentence, label in data[i: i+batch_size]:
                index_vec = [w2i[w] for w in sentence]
                pad_len = max(0, max_sentence_len - len(index_vec))
                index_vec += [0] * pad_len
                index_vec = index_vec[:max_sentence_len] 
                in_data.append(index_vec)
                labels.append(float(label))
            sent_var = Variable(torch.LongTensor(in_data))
            if use_cuda: sent_var = sent_var.cuda()

            target_var = Variable(torch.Tensor(labels).unsqueeze(1))
            if use_cuda: target_var = target_var.cuda()
            optimizer.zero_grad()
            probs = model(sent_var)
            loss = F.binary_cross_entropy(probs, target_var)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.data[0]
        losses.append(epoch_loss)
        
    return model, losses

def test(model, data, n_test, min_sentence_len):
    model.eval()
    loss = 0
    correct = 0
    for sentence, label in data[:n_test]:
        if len(sentence) < min_sentence_len:  # to0 short for CNN's filter
            continue
        index_vec = [w2i[w] for w in sentence]
        sent_var = Variable(torch.LongTensor([index_vec]))
        if use_cuda: sent_var = sent_var.cuda()
        out = model(sent_var)
        score = out.data[0][0]
        pred = 1 if score > .5 else 0
        if pred == label:
            correct += 1
        loss += math.pow((label-score), 2)
        
out_ch = 100
embd_size = 64
batch_size = 16
n_epoch = 5
filter_variations = [[1]]

for fil in filter_variations:
    model = Net(vocab_size, embd_size, out_ch, fil)
    model, losses = train(model, train_data, batch_size, n_epoch)
    test(model, test_data, len(test_data), max(fil))

