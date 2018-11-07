#! /bin/env python
# -*- coding: utf-8 -*-

from keras.layers.core import Activation,Dense
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
import nltk
import collections
import numpy as np

print '1. 数据准备...'
# 句子最大长度
maxlen = 0
# 词频
word_freqs = collections.Counter()
# 样本数
num = 0

with open('./data/train_data.txt','r+') as train_f:
    for line in train_f:
        label,sentence = line.strip().split('\t')
        words = nltk.word_tokenize(sentence.lower())

        if len(words) > maxlen:
            maxlen = len(words)

        for word in words:
            word_freqs[word] += 1
        
        num += 1
'''
共有样本数: 7086
最大句子长度是: 42
单词数目为: 2329
'''
print '共有样本数:',num
print '最大句子长度是:',maxlen
print '单词数目为:',len(word_freqs)

print '2. 构造词向量...'
# 保留训练数据中按词频从大到小排序后的前2000个单词
MAX_FEATURES = 2000
MAX_SENTENCE_LENGTH = 40

# 建立lookup tables用于单词和数字的转换
vocab_size = min(MAX_FEATURES,len(word_freqs)) + 2
## 这个方法会不会太low了
word2index = {x[0]:i+2 for i,x in enumerate(word_freqs.most_common(MAX_FEATURES))}
# 填充和伪单词
word2index['PAD'] = 0
word2index['UNK'] = 1
index2word = {v:k for k,v in word2index.items()}

print '3. 将句子转化为数字序列，并将长度统一...'
X = np.empty(num,dtype=list)
y = np.zeros(num)

index = 0
with open('./data/train_data.txt','r') as f:
    for line in f:
        label,sentence = line.strip().split('\t')
        words = nltk.word_tokenize(sentence.lower())
        seq = []

        for word in words:
            if word in word2index.keys():
                seq.append(word2index[word])
            else:
                seq.append(word2index['UNK'])
        X[index] = seq
        y[index] = int(label)
        index += 1

X = sequence.pad_sequences(X,maxlen=MAX_SENTENCE_LENGTH)

print '4. 划分数据...'
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

print '5. 构建网络...'
EMBEDDING_SIZE = 128
HIDDEN_LAYER_SIZE = 64

model = Sequential()
model.add(Embedding(vocab_size,EMBEDDING_SIZE,input_length=MAX_SENTENCE_LENGTH))
model.add(LSTM(HIDDEN_LAYER_SIZE,dropout=0.2,recurrent_dropout=0.2))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

print '6. 训练数据...'
BATCH_SIZE = 32
NUM_EPOCH = 10
model.fit(X_train,y_train,batch_size=BATCH_SIZE,epochs=NUM_EPOCH,validation_data=(X_test,y_test))

score,acc = model.evaluate(X_test,y_test,batch_size=BATCH_SIZE)
print 'Test score: %.3f, accuary: %.3f' %(score,acc)
# print '{%s}     {%s}      {%s}' %('预测'，'真实','句子')

for i in range(5):
    idx = np.random.randint(len(X_test))
    xtest = X_test[idx].reshape(1,40)
    ylable = y_test[idx]
    ypred = model.predict(xtest)[0][0]
    sent = ' '.join([index2word[x] for x in xtest[0] if x!=0])
    print '%d,%d,%s' %(int(round(ypred)),int(label),sent)

print '7. 做个小小预测...'
INPUT_SENTENCE = ['I love reading.','You are so boring.']
XX = np.empty(len(INPUT_SENTENCE),dtype=list)
i = 0

for sentence in INPUT_SENTENCE:
    words = nltk.word_tokenize(sentence.lower())
    seq = []
    for word in words:
        ## 依靠词典？？？
        if word in word2index.keys():
            seq.append(word2index[word])
        else:
            seq.append(word2index['UNK'])
    XX[i] = seq
    i += 1

XX = sequence.pad_sequences(XX,maxlen=MAX_SENTENCE_LENGTH)
labels = [int(round(x[0])) for x in model.predict(XX)]

label2word = {1:'积极',0:'消极'}

for i in range(len(INPUT_SENTENCE)):
    print '%s:%s' %(label2word[labels[i]],INPUT_SENTENCE[i])





