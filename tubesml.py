# -*- coding: utf-8 -*-
"""TubesML.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ZrTeX48NseVmIAaWEsdSmvLK4DWe9iI7
"""

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


PATH = 'https://raw.githubusercontent.com/Jcharis/Python-Machine-Learning/master/Gender%20Classification%20With%20%20Machine%20Learning/names_dataset.csv'
df = pd.read_csv(PATH)

df.head

df.dtypes

df.isnull().isnull().sum()

df_names = df

df_names.sex.replace({'F':0,'M':1}, inplace=True)

df_names.sex.unique()

X = df_names['name']

cv = CountVectorizer()
X = cv.fit_transform(X)

cv.get_feature_names_out()

y = df_names.sex
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

# Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train,y_train)

# Sample1 Prediction
sample_name = ["Mary"]
vect = cv.transform(sample_name).toarray()

# Female is 0, Male is 1
clf.predict(vect)

# A function to do it
def genderpredictor(a):
    test_name = [a]
    vector = cv.transform(test_name).toarray()
    if clf.predict(vector) == 0:
        print("Female")
    else:
        print("Male")

genderpredictor("Martha")

namelist = ["Obi","Eimi","Femi","Masha"]
for i in namelist:
    print(genderpredictor(i))

# By Analogy most female names ends in 'A' or 'E' or has the sound of 'A'
def features(name):
    name = name.lower()
    return {
        'first-letter': name[0], # First letter
        'first2-letters': name[0:2], # First 2 letters
        'first3-letters': name[0:3], # First 3 letters
        'last-letter': name[-1],
        'last2-letters': name[-2:],
        'last3-letters': name[-3:],
    }

# Vectorize the features function
features = np.vectorize(features)
print(features(["Anna", "Hannah", "Peter","John","Vladmir","Mohammed"]))

# Extract the features for the dataset
df_X = features(df_names['name'])
df_y = df_names['sex']

from sklearn.feature_extraction import DictVectorizer
 
corpus = features(["Mike", "Julia"])
dv = DictVectorizer()
dv.fit(corpus)
transformed = dv.transform(corpus)
print(transformed)

dv.get_feature_names_out()

# Train Test Split
dfX_train, dfX_test, dfy_train, dfy_test = train_test_split(df_X, df_y, test_size=0.3, random_state=42)

dfX_train

dv = DictVectorizer()
dv.fit_transform(dfX_train)

# Model building Using DecisionTree

from sklearn.tree import DecisionTreeClassifier
 
dclf = DecisionTreeClassifier()
my_xfeatures =dv.transform(dfX_train)
dclf.fit(my_xfeatures, dfy_train)

# Build Features and Transform them
sample_name_eg = ["Alex"]
transform_dv =dv.transform(features(sample_name_eg))

vect3 = transform_dv.toarray()

# Predicting Gender of Name
# Male is 1,female = 0
dclf.predict(vect3)

if dclf.predict(vect3) == 0:
    print("Female")
else:
    print("Male")

# Second Prediction With Nigerian Name
name_eg1 = ["Chioma"]
transform_dv =dv.transform(features(name_eg1))
vect4 = transform_dv.toarray()
if dclf.predict(vect4) == 0:
    print("Female")
else:
    print("Male")

# A function to do it
def genderpredictor1(a):
    test_name1 = [a]
    transform_dv =dv.transform(features(test_name1))
    vector = transform_dv.toarray()
    if dclf.predict(vector) == 0:
        print("Female")
    else:
        print("Male")

random_name_list = ["Raisa","Alfa","Chioma","Vitalic","Clairese","Chan"]

for n in random_name_list:
    print(genderpredictor1(n))

## Accuracy of Models Decision Tree Classifier Works better than Naive Bayes
# Accuracy on training set
print(dclf.score(dv.transform(dfX_train), dfy_train)) 

# Accuracy on test set
print(dclf.score(dv.transform(dfX_test), dfy_test))