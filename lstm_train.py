#! /bin/env python
# -*- coding: utf-8 -*-
'''
训练网络
保存模型
'''
import pandas as pd
import numpy as np
import jieba
import sys
import yaml
import keras

from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary

from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.layers.embeddings import Embedding
from keras.models import Sequential
from keras.models import model_from_yaml
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense,Dropout,Activation

np.random.seed(1337)
sys.setrecursionlimit(1000000)

# 设定参数
cpu_count = multiprocessing.cpu_count()
vocab_dim = 100
n_iterations = 1
n_exposures = 10
window_size = 7
n_epoch = 4
input_length = 100
maxlen = 100
batch_size = 32

def loadfile():
    neg_pd = pd.read_csv('./data/neg.csv',header=None,index_col=None,error_bad_lines=False)
    pos_pd = pd.read_csv('./data/pos.csv',header=None,index_col=None,error_bad_lines=False)
    neu_pd = pd.read_csv('./data/neutral.csv',header=None,index_col=None,error_bad_lines=False)
    combined = np.concatenate((pos_pd[0],neu_pd[0],neg_pd[0]))
    # 积极1 中性0 消极-1
    y = np.concatenate((np.ones(len(pos_pd),dtype=int),np.zeros(len(neu_pd),dtype=int),-1*np.ones(len(neg_pd),dtype=int)))
    # print combined[:10]
    # print '==='*20
    
    return combined,y

# 分词并去掉换行符 
def tokenizer(text):
    text = [jieba.lcut(sentence.replace('\n','')) for sentence in text]
    
    return text

# 创建词语字典，并返回每个词语的索引，词向量以及每个句子所对应的词语索引
def word2vec_train(combined):
    model = Word2Vec(
        size=vocab_dim,
        min_count=n_exposures,
        window=window_size,
        workers=cpu_count,
        iter=n_iterations
    )
    model.build_vocab(combined)
    model.train(combined,total_examples=len(combined),epochs=model.iter)
    model.save('./model/Word2Vec_model.pkl')
    index_dict,word_vectors,combined = create_dictionaries(model=model,combined=combined)

    return index_dict,word_vectors,combined

# 创建每个词的索引，词向量，转换训练和测试字典
def create_dictionaries(model=None,combined=None):
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),allow_update=True)
       
        # 所有频数超过20的词索引
        word2index = {v:k+1 for k,v in gensim_dict.items()}
        word2vec = {word:model[word] for word in word2index.keys()}

        # 实现单词转换成索引
        def parse_dataset(combined):
            data = []
            for sentence in combined:
                new_text = []
                for word in sentence:
                    try:
                        new_text.append(word2index[word])
                    except:
                        new_text.append(0)
                data.append(new_text)
            return data

        combined = parse_dataset(combined)
        combined = sequence.pad_sequences(combined,maxlen=maxlen)
        # (21088, 100)
        # print combined.shape

        return word2index,word2vec,combined
    else:
        print 'No data provided...'

def get_data(index_dict,word_vectors,combined,y):
    # 所有单词的索引数，频数小于10的为0，所以+1
    n_symbols = len(index_dict) + 1
    embedding_weights = np.zeros((n_symbols,vocab_dim))

    for word,index in index_dict.items():
        embedding_weights[index,:] = word_vectors[word]

    x_train,x_test,y_train,y_test = train_test_split(combined,y,test_size=0.2)
    y_train = keras.utils.to_categorical(y_train,num_classes=3)
    y_test = keras.utils.to_categorical(y_test,num_classes=3)
    

    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test

def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
    print 'Defining a Simple Keras MOdel...'
    model = Sequential()
    model.add(Embedding(
        output_dim = vocab_dim,
        input_dim = n_symbols,
        mask_zero = True,
        weights = [embedding_weights],
        input_length = input_length
    ))
    model.add(LSTM(activation='tanh',output_dim=50))
    model.add(Dropout(0.5))
    # Dense=>全连接层,输出维度=3
    model.add(Dense(3,activation='softmax'))
    model.add(Activation('softmax'))

    print 'Compiling the Model...'
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    print 'Training...'
    model.fit(x_train,y_train,batch_size=batch_size)
    print 'Evaluate...'
    score = model.evaluate(x_test,y_test,batch_size=batch_size)
    # Test Score: [1.0909853669465464, 0.4253200569555283]
    print 'Test Score:',score

    yaml_string = model.to_yaml()
    with open('./model/lstm.yaml','w') as outfile:
        outfile.write(yaml.dump(yaml_string,default_flow_style=True))
    model.save_weights('./model/lstm.h5')
    
    print 'All Finished!'

print 'Loading Data...'
combined,y = loadfile()
# print combined.shape[0],y.shape[0]

print 'Tokenising...'
combined = tokenizer(combined)
# print combined[:3]

print 'Training a word2vec model...'
index_dict,word_vectors,combined = word2vec_train(combined)

print 'Setting up Arrays for Keras Embedding Layer...'
n_symbols,embedding_weights,x_train,y_train,x_test,y_test = get_data(index_dict,word_vectors,combined,y)

print 'x_train.shape and y_train.shape:'
print x_train.shape,y_train.shape

train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test)
