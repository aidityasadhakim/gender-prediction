import pandas as pd
import numpy as np
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pymongo import MongoClient
from fastapi.middleware.cors import CORSMiddleware  # type: ignore

client = MongoClient(os.getenv('MONGO_SRC'))
gender_data = client.gender_prediction.gender_data

df_names = pd.DataFrame(list(gender_data.find({},{"_id":0})))

df_names.sex.replace({'F':0,'M':1}, inplace=True)

X = df_names['name']

cv = CountVectorizer()
X = cv.fit_transform(X)

y = df_names.sex
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

# Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train,y_train)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

@app.get("api/v1/print")
def print():
    return {
        "data": "Hello World"
    }

@app.get("/api/v1/predict")
# A function to do it
def genderpredictor(name):
    test_name = [name]
    vector = cv.transform(test_name).toarray()
    if clf.predict(vector) == 0:
        return {
            "gender" : "female"
        }
    else:
        return {
            "gender" : "male"
        }

genderpredictor("Martha")
