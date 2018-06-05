# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 14:22:55 2018

@author: e620934 
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

#get and parser html page
#1
import urllib .request, urllib.parse, urllib.error
from bs4 import BeautifulSoup

#url="https://en.wikipedia.org/wiki/Boston"
url="https://www.google.com/?gws_rd=ssl"
request=Request(url)
response=urlopen(request)
html=response.read()
Soup=BeautifulSoup(html)
Soup.title
Soup.get_text()
for link in Soup.find_all('a'):
    print(link.get('href'))

#2
url=input('')
html=urllib.request.urlopen(url).read()
Soup=BeautifulSoup(html，'html.parser')
tags=Soup('a')
for tag in tags:
    print(tag.get('href',None))
    
    
#3
     
fhand=urllib.request.urlopen("https://en.wikipedia.org/wiki/Boston")  # make socket call, but simpler
for line in fhand:
    print(line.decode().strip()) # but didn't get header



#Titanic Kaggle project

from IPython import get_ipython
get_ipython().magic('reset -sf')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
os.chdir('H:\\Others\\data\\Projects\\Titanic')
os.getcwd()
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
train.head()
train.sample()
test.info()
test.describe()
train.info()
train.shape
train.dtypes
train.describe()
train.values
train.columns


#univariate analysis
#embarked vs survived
#==============================================================================
# 
# train.groupby('Embarked')['Survived'].mean()
# #==============================================================================
# # Embarked
# # C    0.553571
# # Q    0.389610
# # S    0.336957
# #==============================================================================
# #Pclass 
# train.groupby('Pclass')['Survived'].mean()
# #==============================================================================
# # Pclass
# # 1    0.629630
# # 2    0.472826
# # 3    0.242363
# #==============================================================================
# # sex
# #train.groupby('Sex')['Survived'].mean()
# train.groupby('Sex').Survived.mean()
# #==============================================================================
# # Sex
# # female    0.742038
# # male      0.188908
# #==============================================================================
# 
# 
# #==============================================================================
# # drop_ind=train[train.Sex.isin (['female'])].index
# # train.shape
# # zzz=train.drop(drop_ind)
# # zzz.head(10)
# # zzz.shape
# # train.head(10)
# # print(drop_ind)
# #==============================================================================
# space=np.linspace(min(train.Age),max(train.Age)).reshape(-1,1)
# space.shape #50,1
# 
# plt.plot([0,1],[0,1],'k--')
# plt.show()
# train.mean()
# 
# #google and read wiki etc: what information could be helpful for this problem
# #women children first, but could fall in the water from lifeboats
# #uneven treatment of three classes
# #passenger vs crew?
# 
# # Dig into missing of age and cabin: orders any patterns
# train.isnull().sum(axis=1)
# train[train['Age'].isnull()]['Survived'].describe()
# train[train['Cabin'].isnull()]['Survived'].describe()
# test[test['Age'].isnull()]
# train['Sex'].describe()
# 
# train.groupby('Embarked').Survived.mean()
# train.groupby('SibSp')['Survived'].mean()
# train.groupby('Parch')['Survived'].mean()
# train.groupby(['Pclass','Embarked'])['Survived'].mean()
# train.groupby(['Parch'])['Survived'].mean()
# train['Age_bins']=pd.qcut(train['Age'],10)
# train.groupby('Age_bins')['Survived'].mean()
# train.nunique()
# train.shape[0]
# train.groupby('Survived').Age.nunique()
# train['Age_bins2']=pd.qcut(train['Age'],20)
# train.groupby('Age_bins2').Survived.mean().plot(style='--')
# train.groupby('Age_bins').Survived.mean().plot(style='--')
# train['Age_bins3']=pd.cut(train['Age'],bins=10)
# train['Age_bins3'].unique()
# train.groupby('Age_bins3').Survived.mean().plot(style='--')
# train.groupby('Age_bins3').Survived.mean()
# 
# #make plots-might save time than look at features
# #make grid sub-plots
# fig, axes = plt.subplots(1, 7, figsize=(30,4))
# axes[0].hist(train['Age'].dropna(),alpha=0.9, color='blue')
# axes[0].grid(True)
# axes[0].set_title('Age')
# bins=6
# mu=train['Age'].mean()
# sigma=train['Age'].std()
# #y = norm.pdf(bins, mu, sigma)
# #plt.plot(bins, y, 'r--')
# #adjust to prevent label clipping
# plt.subplots_adjust(left=0.15)
# plt.show()
# 
# #another way for subplots
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.hist(train['Age'].dropna(), ls='dashed', alpha = 0.5, lw=3, color= 'b')
# ax.hist(train['Pclass'].dropna(), ls='dotted', alpha = 0.5, lw=3, color= 'r')
# #ax.set_xlim(-0.5, 1.5)
# #ax.set_ylim(0, 7)
# plt.show()
# 
# #matrix plot
# fig=plt.figure(figsize=(100,15))
# pd.plotting.scatter_matrix(train)
# correl=train.corr()
# plt.matshow(correl)
# 
# 
#==============================================================================

train_response = train['Survived']
str(train_response)
train_response.head(50)
train.drop('Survived', axis=1,inplace=True)
# create features and do baseline: notebook 3, random tree, important features
#train['Age_bins3']=pd.cut(train['Age'],bins=10)
combine=[train, test]
for dataset in combine:
  dataset['PassengerId']=dataset['PassengerId'].astype('str')
  dataset.set_index('PassengerId', inplace=True)
  dataset['Embarked_f']=dataset['Embarked'].fillna(value='C')
  dataset['Fare']=dataset['Fare'].fillna(dataset.Fare.mean())
   
  dataset.loc[dataset.Age<8.378, 'Age_factor']=1
  dataset.loc[(dataset.Age <16.336) & (dataset.Age>=8.378),'Age_factor']=2
  dataset.loc[(dataset.Age<24.294) & (dataset.Age>=16.336),'Age_factor']=3
  dataset.loc[(dataset.Age<32.252) & (dataset.Age>=24.294),'Age_factor']=4
  dataset.loc[(dataset.Age<40.21) & (dataset.Age>=32.252),'Age_factor']=5
  dataset.loc[(dataset.Age<48.168) & (dataset.Age>=40.21),'Age_factor']=6
  dataset.loc[(dataset.Age<56.126) & (dataset.Age>=48.168),'Age_factor']=7
  dataset.loc[(dataset.Age<64.084) & (dataset.Age>=56.126),'Age_factor']=8
  dataset.loc[(dataset.Age<72.042) & (dataset.Age>=64.084),'Age_factor']=9
  dataset.loc[dataset.Age>=72.042,'Age_factor']=10
  dataset.loc[dataset.Age.isnull(),'Age_factor']=11

  dataset['Family_size']=dataset['SibSp']+dataset['Parch']+1

  dataset['Fare_per_person']=dataset['Fare']/dataset['Family_size']

 # dataset['Fare_bins']=pd.qcut(train['Fare_per_person'],3)
  dataset.loc[dataset.Fare_per_person <7.775,'Fare_factor']=1
  dataset.loc[(dataset.Fare_per_person <13) & (dataset.Fare_per_person >=7.775),'Fare_factor']=2           
  dataset.loc[dataset.Fare_per_person >=13,'Fare_factor']=3          

 
  dataset.loc[dataset['Cabin'].isnull(),'Cabin_Y']="N" 
  dataset.loc[dataset['Cabin'].notnull(),'Cabin_Y']="Y"   
  
  dataset.loc[dataset.Cabin.isnull(), 'Deck']="No"
  dataset.loc[dataset.Cabin.notnull(),'Deck']=dataset.Cabin.str.slice(0,1)

  
#==============================================================================
#   dataset['Sex_factor']=pd.factorize(dataset['Sex'])[0]
#   dataset['Embarked_factor']=pd.factorize(dataset['Embarked_f'])[0]
#   dataset['Cabin_factor']=pd.factorize(dataset['Cabin_Y'])[0]
#   dataset['Deck_factor']=pd.factorize(dataset['Deck'])[0]
#==============================================================================
  dataset.drop(['Name','Age','SibSp','Parch','Ticket','Fare','Cabin','Embarked','Fare_per_person'],axis=1, inplace=True)
    
#==============================================================================
#   dataset['FareFamily']=dataset['Fare_factor']*dataset['Family_size']
#   dataset['AgeFamily']=train['Age_factor']*train['Family_size']
#   dataset['FareFamily']=pd.factorize(dataset['FareFamily'])[0]
#   dataset['AgeFamily']=pd.factorize(dataset['AgeFamily'])[0]
#==============================================================================
  
  

# creat dummies for categoricals  
train_with_dummies= pd.get_dummies(train, columns=['Pclass','Sex','Embarked_f','Age_factor','Family_size','Fare_factor','Cabin_Y','Deck'])
test_with_dummies= pd.get_dummies(test, columns=['Pclass','Sex','Embarked_f','Age_factor','Family_size','Fare_factor','Cabin_Y','Deck'])
test_with_dummies['Deck_T']=0  



# fit some base models
#logistic regression

from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(train_with_dummies,train_response)
pred=logreg.predict(test_with_dummies)
logreg.score(train_with_dummies, train_response)

from sklearn.model_selection import cross_val_score
cv_result=cross_val_score(logreg, train_with_dummies, train_response, cv=5)

#n-near neighbors
from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=9)
knn.fit(train_with_dummies,train_response)
pred=knn.predict(test_with_dummies)
knn.score(train_with_dummies, train_response)

cv_result=cross_val_score(knn, train_with_dummies, train_response, cv=5)



from sklearn.model_selection import GridSearchCV
param_grid={'n_neighbors':np.arange(1,50)}
knn=KNeighborsClassifier()
knn_cv=GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(train_with_dummies,train_response)
knn_cv.best_params_
knn_cv.best_score_



#naive bayes
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(train_with_dummies,train_response)
pred=gnb.predict(train_with_dummies)
gnb.score(train_with_dummies, train_response)
from sklearn.metrics import accuracy_score
accuracy_score(train_response,pred)

from sklearn.model_selection import cross_val_score
cv_result=cross_val_score(gnb, train_with_dummies, train_response, cv=5)


#svm
from sklearn.svm import SVC
lsvm=SVC(kernel='linear')
lsvm.fit(train_with_dummies,train_response)
pred=lsvm.predict(test_with_dummies)
lsvm.score(train_with_dummies, train_response)


rsvm=SVC(kernel='rbf')
rsvm.fit(train_with_dummies,train_response)
pred=rsvm.predict(test_with_dummies)
rsvm.score(train_with_dummies, train_response)

#best score is the mean cross validation score for the best parameters
from sklearn.model_selection import GridSearchCV
def svc_param_selection(X, y, nfolds):
  Cs = [0.001, 0.01, 0.1, 1, 10]
  gammas = [0.001, 0.01, 0.1, 1]
  param_grid = {'C': Cs, 'gamma' : gammas}
  grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=nfolds)
  grid_search.fit(X, y)
  
  grid_search.best_score_
  return {'Best score':grid_search.best_score_, 'Best params':grid_search.best_params_}

svc_param_selection(train_with_dummies, train_response, 5)
svc_param_selection(train_with_dummies, train_response, 6)
svc_param_selection(train_with_dummies, train_response, 7) 

#libsvm


#random forest

from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(train_with_dummies,train_response)
pred=rf.predict(test_with_dummies)
rf.score(train_with_dummies, train_response)

from sklearn.model_selection import GridSearchCV
#==============================================================================
# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=2, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
#             oob_score=False, random_state=0, verbose=0, warm_start=False)
# 
#==============================================================================



# how to choose scoring for GridSearchCV see: http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

bootstrap=['True', 'False']
max_depth=[10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None]
n_estimators=[10, 50, 100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
min_samples_split=[2,5,10]
min_samples_leaf=[1,2,4]
max_features=['auto', 'sqrt']

param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(train_with_dummies,train_response)
grid_search.best_score_
grid_search.best_params_  



#neura network

#bagging

from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
bagging.fit(train_with_dummies,train_response)
pred=bagging.predict(train_with_dummies)
bagging.score(train_with_dummies, train_response)

from sklearn.metrics import accuracy_score
accuracy_score(train_response,pred)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(bagging,train_with_dummies,train_response )
scores.mean()


#gradient boosting

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1)
gb.fit(train_with_dummies,train_response)
pred=gb.predict(train_with_dummies)
gb.score(train_with_dummies, train_response)

from sklearn.metrics import accuracy_score
accuracy_score(train_response,pred)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(gb,train_with_dummies,train_response )
scores.mean() 




#adaboosting
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier
adb = AdaBoostClassifier(n_estimators=100)
adb.fit(train_with_dummies,train_response)
pred=adb.predict(train_with_dummies)
adb.score(train_with_dummies, train_response)

from sklearn.metrics import accuracy_score
accuracy_score(train_response,pred)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(adb,train_with_dummies,train_response )
scores.mean() 




#xgboost
from sklearn.model_selection import cross_val_score
import xgboost as xgb

xgboost=xgb.XGBClassifier(objective='binary:logistic', n_estimators=10)
xgboost.fit(train_with_dummies,train_response)
xgboost.score(train_with_dummies, train_response)

#stacking



#submission
test_with_dummies['Survived']=pred
test_with_dummies=test_with_dummies['Survived']
test_with_dummies.to_csv('gender_submission.csv', header=True)







#compare train and test, figure out validation plan. lb probing
