#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 14:31:05 2018

@author: carolinalissack
"""

from sklearn.decomposition import NMF
import pandas as pd
from scipy.sparse import csr_matrix

df = pd.read_csv('wiki_source.csv', index_col=0)
articles = csr_matrix(df.transpose())
titles = list(df.columns)

model = NMF(n_components=6)
model.fit(articles)

nmf_features = model.transform(articles)

df = pd.DataFrame(nmf_features, index=titles)

article = df.loc['Anne Hathaway']

#Dot products
similarities = df.dot(article)

# Display those with the largest cosine similarity
print(similarities.nlargest())
