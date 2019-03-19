# coding: utf-8
from sklearn.metrics import mean_squared_error
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy as np
import pandas as pd
import os

y_var = 'ret'
final_y_pred = np.load("pred.npy")

raw_data = pd.read_csv(os.path.join("data", "prova.csv"))
data = scaler.transform(raw_data)
data = pd.DataFrame(data, columns=raw_data.columns)

df = data[[y_var]].copy()
df = df.iloc[-len(final_y_pred):].copy()

try:
    df['pred'] = final_y_pred
    s1 = precision_score(df[y_var], df['pred'], average='weighted') * 100
    s2 = recall_score   (df[y_var], df['pred'], average='weighted') * 100
    s3 = f1_score       (df[y_var], df['pred'], average='weighted') * 100
except:
    df['prob'] = final_y_pred
    df['pred'] = np.where( df['prob']>1/3, 1, np.where( df['prob']<-1/3 ,-1,0)  )
    s1 = precision_score(df[y_var], df['pred'], average='weighted') * 100
    s2 = recall_score   (df[y_var], df['pred'], average='weighted') * 100
    s3 = f1_score       (df[y_var], df['pred'], average='weighted') * 100

print("precision_score: {0:0.2f}%, \nrecall_score: {1:0.2f}%, \nf1_score: {2:0.2f}%".format(s1,s2,s3))