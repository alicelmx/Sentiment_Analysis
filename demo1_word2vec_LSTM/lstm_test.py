#! /bin/env python
# -*- coding: utf-8 -*-
"""
利用之前训练好的模型进行预测
"""
import numpy as np
import jieba
import sys
import yaml

from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
from keras.models import model_from_yaml

np.random.seed(1337)
sys.setrecursionlimit(1000000)

maxlen = 100

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

def input_transform(string):
    words = jieba.lcut(string)
    words = np.array(words).reshape(1,-1)
    model = Word2Vec.load('./model/Word2Vec_model.pkl')
    _,_,combined = create_dictionaries(model,words)
    return combined

def lstm_predict(data):
    print 'Loading Model...'
    with open('./model/lstm.yaml','r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)

    print 'Loading Weights...'
    model.load_weights('./model/lstm.h5')
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    data.reshape(-1,1)
    # print data
    
    result = model.predict_classes(data)
    # print result

    if result[0] == 1:
        print 'positive'
    elif result[0] == 0:
        print 'neural'
    else:
        print 'negative'

def main():
    stringlist = [
        '酒店的环境非常好，价格也便宜，值得推荐',
        '手机质量太差了，傻逼店家，赚黑心钱，以后再也不会买了',
        "这是我看过文字写得很糟糕的书，因为买了，还是耐着性子看完了，但是总体来说不好，文字、内容、结构都不好",
        "虽说是职场指导书，但是写的有点干涩，我读一半就看不下去了！",
        "书的质量还好，但是内容实在没意思。本以为会侧重心理方面的分析，但实际上是婚外恋内容。",
        "不是太好",
        "不错不错",
        "真的一般，没什么可以学习的"
    ]
    
    for string in stringlist:
        print string
        data = input_transform(string)

        lstm_predict(data)

if __name__ == '__main__':
    main()