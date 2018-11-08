#! /bin/env python
# -*- coding: utf-8 -*-
import pandas as pd

neg_pd = pd.read_csv('neg.csv',header=None,index_col=None,error_bad_lines=False)
pos_pd = pd.read_csv('pos.csv',header=None,index_col=None,error_bad_lines=False)
neu_pd = pd.read_csv('neutral.csv',header=None,index_col=None,error_bad_lines=False)

'''
积极： 8030
消极： 8703
中性： 4355
'''
print '积极：',pos_pd.shape[0]
print '消极：',neg_pd.shape[0]
print '中性：',neu_pd.shape[0]
