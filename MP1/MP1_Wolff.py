#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 13:13:50 2022

@author: hillarywolff
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

PATH = r"/Users/hillarywolff/Documents/GitHub/machine_learning/MP1/"
 


def read_data(fname):
    df = pd.read_csv(os.path.join(PATH, fname)) 
    return df

fname = 'usa_00001.csv'

df = read_data(fname)

crosswalk_fname = 'PPHA_30545_MP01-Crosswalk.csv'

crosswalk = read_data(crosswalk_fname)
crosswalk = crosswalk.set_index('educd').T
crosswalk = crosswalk.to_dict('list')


df['EDUCDC'] = df['EDUCD']
df = df.replace({'EDUCDC':crosswalk})

df['hsdip'] = np.where((df['EDUC']==62)| 
                       (df['EDUCD']==63)|
                       (df['EDUCD']==64), 1, 0)

df['coldip'] = np.where((df['EDUCD']==101)|
                        (df['EDUCD']==114)|
                        (df['EDUCD']==115)|
                        (df['EDUCD']==115)|
                        (df['EDUCD']==116), 1, 0)

df['White'] = np.where(df['RACE'] == 1, 1, 0)

df['Black'] = np.where(df['RACE'] == 2, 1, 0)

df['hispanic'] = np.where((df['HISPAN']==1)|
                          (df['HISPAN']==2)|
                          (df['HISPAN']==3)|
                          (df['HISPAN']==4), 1, 0)

df['married']= np.where((df['MARST']==1)|
                        (df['MARST']==2), 1, 0)

df['female']= np.where((df['SEX']==2), 1, 0)

df['vet'] =np.where((df['VETSTAT']==2), 1, 0)

df['EDUxHISPAN'] = df['EDUCDC']*df['hispanic']
df['EDUxHSDIP'] = df['EDUCDC']*df['hsdip']
df['EDUxCOLDIP'] = df['EDUCDC']*df['coldip']
df['EDUxWHITE'] = df['EDUCDC']*df['White']
df['EDUxBLACK'] = df['EDUCDC']*df['Black']
df['EDUxMARRIED']= df['EDUCDC']*df['married']
df['EDUxFEMALE'] = df['EDUCDC']*df['female']
df['EDUxVET'] = df['EDUCDC']*df['vet']

df['age_sq'] = np.power(df['AGE'], 2)
df['INCWAGE_log'] = np.log(df['INCWAGE'])

describe_cols = ['EDUxHISPAN', 'EDUxHSDIP', 'EDUxCOLDIP', 'EDUxWHITE', 'EDUxBLACK',
            'EDUxMARRIED', 'EDUxFEMALE', 'EDUxVET', 'vet', 'female', 'Black',
            'White', 'hispanic', 'coldip', 'hsdip', 'EDUCDC', 'age_sq', 'AGE', 
            'INCWAGE','INCWAGE_log', 'NCHILD', 'YEAR']

reg_cols = ['EDUCDC', 'female', 'AGE', 'age_sq', 'White', 'Black', 
            'hispanic', 'married', 'NCHILD', 'vet']
reg_cols = ' + '.join(reg_cols)

describe_df=pd.DataFrame()
for col in describe_cols:
    describe_df.append(df[col].describe())

plt.scatter(df['EDUCDC'], df['INCWAGE_log'])
plt.title('ln(wage) vs. Educational Attainment')
plt.ylabel('ln(wage) ($10,000)')
plt.xlabel('Education level')




result = smf.ols('INCWAGE_log ~ {}'.format(reg_cols), data=df).fit().summary()
results_as_html = result.tables[1].as_html()
reg_df = (pd.read_html(results_as_html, header=0, index_col=0)[0]).reset_index()





















































