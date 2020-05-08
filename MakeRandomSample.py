import csv
import random
import re

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words("english"))
ps = PorterStemmer()

#cleaner
clean1 = re.compile("["
        u"\u0021-\u0040"
        u"\u005b-\u0060" 
        u"\u007b-\u007e"
        u"\u0000-\u001f"                     "]+", re.UNICODE)
    

#get only ID, review text
with open('OnlyReviews.csv', 'r', newline='') as readrnd:
    reader = csv.reader(readrnd)
    with open('OnlyReviewsCleaned.csv', 'w', newline='') as writernd:
         writer = csv.writer(writernd)
         for r in reader:
             words = word_tokenize(clean1.sub(r'', r[1].lower()).replace('   ',''))
             words_filtered = [w for w in words if not w in stop_words]
             
             stemmed = []
             for w in words_filtered:
                 stemmed.append(ps.stem(w))
              
             writer.writerow((r[0],stemmed))



#get only review text CLEANED PROTOTYP
with open('datasettest.csv', 'r', newline='') as readrnd:
    reader = csv.reader(readrnd)
    with open('OnlyReviewsCleanedfinal.csv', 'w', newline='') as writernd:
         writer = csv.writer(writernd)
         for r in reader:
             words = word_tokenize(clean1.sub(r'', r[2].lower()).replace('   ',''))
             words_filtered = [w for w in words if not w in stop_words]
             
             stemmed = []
             for w in words_filtered:
                 stemmed.append(ps.stem(w))
              
             writer.writerow((stemmed,))
    



#get random sample
with open('OnlyReviews.csv', 'r', newline='') as readfile:
    reader = csv.reader(readfile)
    lines = [line for line in readfile]
    random_choice = random.sample(lines, 2500)
    
    with open('Random_Selection.csv', 'w', newline='') as writefile:
        writefile.write("\n".join(random_choice).replace('\n',''))
        
        


#clean random sample Symbols, tokenize, stopwords, stemming
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
    
    