import pandas as pd
import numpy as np
import textstat

import math
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import pearsonr

import statsmodels.api as sm
from statsmodels.stats import diagnostic as diag
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

dataset = pd.read_csv('datasettest.csv')

CompanyMeans = dataset.groupby('Company')[['Status', 
                                           'Rating', 
                                           'Useful',
                                           'Verified',
                                           'User Reviews',
                                           'Wordcount Review',
                                           'Readability',
                                           'Diagnosticity',
                                           'Wordcount Response',
                                           'Readability Response',
                                           'Response Speed in Days'
                                           #'Gunning'
                                                     ]].mean()


CompanyMeans['ReviewCount']  = dataset.groupby('Company').size()
CompanyMeans['ResponseCount']  = dataset.groupby('Company')['Response Text'].count()
CompanyMeans['ResponseFrequency'] = dataset.groupby('Company')['Response Text'].count() / dataset.groupby('Company').size() 

hhhh = CompanyMeans.describe()


OverallMeans = dataset[['Status',
                        'Rating', 
                        'Useful',
                        'Verified',
                        'User Reviews',
                        'Wordcount Review',
                        'Readability',
                        'Diagnosticity',
                        'Readability Response',
                        'Response Speed in Days',
                        'Wordcount Response']].mean()
                        

OverallMeans['ReviewCount']  = dataset['Rating'].count()
OverallMeans['ResponseCount']  = dataset['Response Text'].count()
OverallMeans['ResponseFrequency'] = dataset['Response Text'].count() / dataset['Rating'].count()
#OverallMeans['Gunning'] = text['Gunning'].mean()




OverallDescr = dataset[['Status',
                        'Rating', 
                        'Useful',
                        'Verified',
                        'User Reviews',
                        'Wordcount Review',
                        'Readability',
                        'Diagnosticity',
                        'Wordcount Response',
                        'Readability Response',
                        'Response Speed in Days']].describe()






def index_marks(nrows, chunk_size):
    return range(chunk_size, math.ceil(nrows / chunk_size) * chunk_size, chunk_size)

def split(dataset, chunk_size):
    indices = index_marks(dataset.shape[0], chunk_size)
    return np.split(dataset, indices)

chunks = split(dataset, 20)

ChunkMeans = []
Companies = []
MRFreq = []
ReviewCount = []

for i in chunks:
    #print(i['Response Text'].count() / 20)
    
    ChunkMeans.append(i[['Wordcount Review',
                         'Useful',
                         'Readability',
                         'Diagnosticity',
                         'Wordcount Response',
                         'Response Speed in Days',
                         'Readability Response',
                         'Rating', 
                         'Status',
                         'Verified',
                         'User Reviews',

                        ]].mean())
   
    Companies.append(i['Company'].values[1])
    MRFreq.append(i['Response Text'].count() / 20)
    ReviewCount.append(i['Review Count'].values[1])
    
Chunkiboi = pd.DataFrame(ChunkMeans)
Chunkiboi['MRFreq'] = MRFreq
Chunkiboi['Review Count'] = ReviewCount
Chunkiboi['Company'] = Companies

Wordcount_Review_log = np.log(Chunkiboi['Wordcount Review'])
Chunkiboi['Wordcount_Review_log'] = Wordcount_Review_log

Useful_log = (np.log(Chunkiboi['Useful']+0.001))
Chunkiboi['Useful_log'] = Useful_log             


Chunkiboi = Chunkiboi[['Wordcount_Review_log', 'Useful_log', 'Readability', 'Diagnosticity',
 'MRFreq', 'Wordcount Response', 'Response Speed in Days', 'Readability Response', 
 'Rating', 'Status', 'Verified', 'User Reviews', 'Review Count', 'Company']]



Descriptives = Chunkiboi.describe()


Correlations = Chunkiboi.corr()

def calculate_pvalues(df):
    df = df.dropna()._get_numeric_data()
    dfcols = pd.DataFrame(columns=df.columns)
    pvalues = dfcols.transpose().join(dfcols, how='outer')
    for r in df.columns:
        for c in df.columns:
            pvalues[r][c] = round(pearsonr(df[r], df[c])[1], 4)
    return pvalues

pValues = calculate_pvalues(Chunkiboi)


rho = Chunkiboi.corr()
rho = rho.round(2)
pval = calculate_pvalues(Chunkiboi) 
# create three masks
r1 = rho.applymap(lambda x: '{}*'.format(x))
r2 = rho.applymap(lambda x: '{}**'.format(x))
r3 = rho.applymap(lambda x: '{}***'.format(x))
# apply them where appropriate
rho = rho.mask(pval<=0.1,r1)
rho = rho.mask(pval<=0.05,r2)
rho = rho.mask(pval<=0.01,r3)


print(stats.skew(Chunkiboi.drop(['Company'], axis = 1), nan_policy='omit'))
print(stats.skew(Chunkiboi['Wordcount_Review_log']))
print(stats.skew(Chunkiboi['Useful_log']))





#plots = sns.pairplot(Chunkiboi_dropped)




#VIF
X1 =  sm.tools.add_constant(Chunkiboi.drop(['Company'], axis = 1).dropna())
series = pd.Series([variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])], index = X1.columns)
print(series)



#Regressions
Chunkiboi_dropped = Chunkiboi.dropna()

#Null models
Xnull = Chunkiboi_dropped.drop(['Wordcount_Review_log', 'Useful_log', 'Readability', 'Diagnosticity',
 'MRFreq', 'Wordcount Response', 'Response Speed in Days', 'Company', 'Review Count'], axis=1)
Xnull = sm.add_constant(Xnull)

Ynull1 = Chunkiboi_dropped[['Wordcount_Review_log']]
modelnull1 = sm.OLS(Ynull1, Xnull).fit() ## sm.OLS(output, input)
#predictionsnull1 = modelnull1.predict(Xnull)
modelnull1.summary()










Xnull = Chunkiboi_dropped.drop(['Wordcount_Review_log', 'Useful_log', 'Readability', 'Diagnosticity',
 'MRFreq', 'Wordcount Response', 'Response Speed in Days','Company'], axis=1)
Xnull = sm.add_constant(Xnull)

Ynull1 = Chunkiboi_dropped[['Wordcount_Review_log']]
modelnull1 = sm.OLS(Ynull1, Xnull).fit() ## sm.OLS(output, input)
#predictionsnull1 = modelnull1.predict(Xnull)
modelnull1.summary()


Ynull2 = Chunkiboi_dropped[['Useful_log']]
modelnull2 = sm.OLS(Ynull2, Xnull).fit() ## sm.OLS(output, input)
#predictions0 = modelnull.predict(Xnull)
modelnull2.summary()

Ynull3 = Chunkiboi_dropped[['Readability']]
modelnull3 = sm.OLS(Ynull3, Xnull).fit() ## sm.OLS(output, input)
#predictions0 = modelnull.predict(Xnull)
modelnull3.summary()

Ynull4 = Chunkiboi_dropped[['Diagnosticity']]
modelnull4 = sm.OLS(Ynull4, Xnull).fit() ## sm.OLS(output, input)
#predictions0 = modelnull.predict(Xnull)
modelnull4.summary()


#MRModels with all chunks and only MRFreq


XMRF = Chunkiboi.drop(['Wordcount_Review_log', 'Useful_log', 'Readability', 'Diagnosticity',
                              'Wordcount Response', 'Response Speed in Days', 'Readability Response', 'Company', 'Review Count'], axis=1)
XMRF = sm.add_constant(XMRF)

YMRF1 = Chunkiboi[['Wordcount_Review_log']]
modelMRF1 = sm.OLS(YMRF1, XMRF).fit() ## sm.OLS(output, input)
#predictionsnull1 = modelnull1.predict(Xnull)
modelMRF1.summary()

YMRF2 = Chunkiboi[['Useful_log']]
modelMRF2 = sm.OLS(YMRF2, XMRF).fit() ## sm.OLS(output, input)
#predictions0 = modelnull.predict(Xnull)
modelMRF2.summary()

YMRF3 = Chunkiboi[['Readability']]
modelMRF3 = sm.OLS(YMRF3, XMRF).fit() ## sm.OLS(output, input)
#predictions0 = modelnull.predict(Xnull)
modelMRF3.summary()

YMRF4 = Chunkiboi[['Diagnosticity']]
modelMRF4 = sm.OLS(YMRF4, XMRF).fit() ## sm.OLS(output, input)
#predictions0 = modelnull.predict(Xnull)
modelMRF4.summary()




#MR model with all MR measures  and only chunks with responses



XMR = Chunkiboi_dropped.drop(['Wordcount_Review_log', 'Useful_log', 'Readability', 'Diagnosticity','Company', 'Review Count'], axis=1)
XMR = sm.add_constant(XMR)

YMR1 = Chunkiboi_dropped[['Wordcount_Review_log']]
modelMR1 = sm.OLS(YMR1, XMR).fit() ## sm.OLS(output, input)
#predictionsnull1 = modelnull1.predict(Xnull)
modelMR1.summary()

YMR2 = Chunkiboi_dropped[['Useful_log']]
modelMR2 = sm.OLS(YMR2, XMR).fit() ## sm.OLS(output, input)
#predictions0 = modelnull.predict(Xnull)
modelMR2.summary()

YMR3 = Chunkiboi_dropped[['Readability']]
modelMR3 = sm.OLS(YMR3, XMR).fit() ## sm.OLS(output, input)
#predictions0 = modelnull.predict(Xnull)
modelMR3.summary()

YMR4 = Chunkiboi_dropped[['Diagnosticity']]
modelMR4 = sm.OLS(YMR4, XMR).fit() ## sm.OLS(output, input)
#predictions0 = modelnull.predict(Xnull)
modelMR4.summary()






















diag.het_goldfeldquandt(modelMR2.resid, modelMR2.model.exog)
diag.het_breuschpagan(modelMR4.resid, modelMR4.model.exog)
diag.het_white(modelMR2.resid, modelMR2.model.exog, retres = False)
diag.(modelMR2.resid, modelMR2.model.exog)


diag.acorr_ljungbox(modelMR2.resid)



