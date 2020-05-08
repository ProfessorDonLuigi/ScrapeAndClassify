import nltk
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn import metrics

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

#clean random sample Symbols, tokenize, stopwords, stemming
stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

with open('Random_Selection.csv', 'r', newline='') as readrnd:
    reader = csv.reader(readrnd)
    with open('Random_Selection_cleaned.csv', 'w', newline='') as writernd:
         writer = csv.writer(writernd)
         for r in reader:
             words = word_tokenize(clean1.sub(r'', r[1].lower()).replace('   ',''))
             words_filtered = [w for w in words if not w in stop_words]
             
             stemmed = []
             for w in words_filtered:
                 stemmed.append(ps.stem(w))
              
             writer.writerow((r[0],stemmed,r[2]))
             
             
             


data = pd.read_csv('Random_Selection_cleaned.csv')


X_train, X_test, y_train, y_test = train_test_split(data['text'], data['diagnosticity'], test_size=0.20)
    
#LinearSVC
pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)),
                     ('chi', SelectKBest(chi2, k=800)),
                     ('clf', LinearSVC(C=1.0, penalty='l1', max_iter=3000, dual=False) )])

#Bernoulli NaiveBayes
pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True)),
                     ('chi', SelectKBest(chi2, k=500)),
                     ('clf', BernoulliNB() )])    
    
#Logistic Regression    
pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True)),
                     ('chi', SelectKBest(chi2, k=1000)),
                     ('clf', LogisticRegression(C=1.0, penalty='l2', max_iter=3000, dual=False, solver='lbfgs') )])    

#Random Forest
pipeline = Pipeline([('vect', TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True)),
                     ('chi', SelectKBest(chi2, k=500)),
                     ('clf', RandomForestClassifier(n_estimators=100) )])  
    
    
    
    
 
#fitting model
model = pipeline.fit(X_train, y_train)

vectorizer = model.named_steps['vect']
chi = model.named_steps['chi']
clf = model.named_steps['clf']


#eval-metrics
accuracy = cross_val_score(model, X_test, y_test, cv=10, scoring='accuracy' )
precision = cross_val_score(model, X_test, y_test, cv=10, scoring='precision' )
recall = cross_val_score(model, X_test, y_test, cv=10, scoring='recall' )
f1score = cross_val_score(model, X_test, y_test, cv=10, scoring='f1' )


#K-Fold Evaluation Metrics in % (means)
print("Accuracy:" + str(accuracy.mean()*100),
      "Precision:" + str(precision.mean()*100),
      "Recall:" + str(recall.mean()*100),
      "F1-score:" + str(f1score.mean()*100))




#run model over all reviews
alldata = pd.read_csv('OnlyReviewsCleanedfinal.csv')

dlist = model.predict(alldata['text'])


    




    