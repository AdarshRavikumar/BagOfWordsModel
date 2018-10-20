# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 00:22:16 2018

@author: Adarsh
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Here we are importing the file of type tsv instead of csv because csv is basically a comma seperated values
# that is in csv columns are seperated using comma's
# there is a chance that the reviews may contain comma , so we use tsv (tab seperated values)
dataset=pd.read_csv('Restaurant_Reviews.tsv',delimiter='\t',quoting=3)
# here quoting =3, means we quote none of the reviews


# Cleaning the text of reviews

import re

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

# here in the above line
# the hat ^ tells not to remove character that are a-z and A-z
# basically the first parameter was to write what u want to remove 
#Since we dont want anything except a-z , we just use ^ and write it as above
cleaned_reviews=[]

for i in range(0,len(dataset)):

    review=re.sub('[^a-zA-Z]',' ',dataset['Review'][i])
    
    review=review.lower()
    
    review=review.split()
    
    # we use stopwards module to remove all irrelavent words like preposition , is , at ,this, which has no impact on review being positive or negative
    
    # review=[words for words in review if words not in stopwords.words('english')]
    
    
    # instead of applying stemming using another loap we will combine stemming and stopwards in one loop
    
    ps=PorterStemmer()
    
    review=[ps.stem(words) for words in review if words not in stopwords.words('english')]
    
    # now convert this list of words to a sentence
    review=' '.join(review)
    
    cleaned_reviews.append(review)
    


# Bag of Words Model

# in this model we take all different words (unique words) from the 1000 reviews and every word and create one column for eah word
#we will have 1000 rows each corresponding to a single review
# each cell has count of how many times that word appeared in the review
# ex: wow love pizza love  
# in the first row cell of wow has entry 1, pizza has entry 1 while love has entry 2 since it appeared twice

#creating bag of Words model
    
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

# in the count vectorizer class there is a parameter alled max_features which keeps only words that appear frequently in reviews in the final sparse matrix

X=cv.fit_transform(cleaned_reviews).toarray()
# we are adding toarray to convert x to matrix of features

y=dataset.iloc[:,1].values

# using Deesion tree classifier


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Fitting classifier to the Training set

from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()

classifier.fit(X_train,y_train)
# Create your classifier here

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc=accuracy_score(y_test,y_pred)








