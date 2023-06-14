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
from fastapi.middleware.cors import CORSMiddleware  # type: ignore

from pydantic import BaseModel

from pymongo import MongoClient
from bson.objectid import ObjectId
from pymongo.server_api import ServerApi

class Person(BaseModel):
    name: str
    sex: str

# Initializing mongodb client
client = MongoClient(os.getenv('MONGO_SRC'), server_api=ServerApi('1'))
gender_data = client.gender_prediction.gender_data
counter_db = client.gender_prediction.counter

# Initializing naive bayes and count vectorizer
clf = MultinomialNB()
cv = CountVectorizer()

def run_model():
    df_names = pd.DataFrame(list(gender_data.find({},{"_id":0})))

    df_names.sex.replace({'F':0,'M':1}, inplace=True)

    X = df_names['name']

    X = cv.fit_transform(X)

    y = df_names.sex
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=42)

    # Naive Bayes Classifier
    clf.fit(X_train,y_train)

run_model()

def update_counter():
    counter_db.update_one({'name': 'counter'}, {'$inc': {'counter': 1}})

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

@app.get("api/v1/print")
def prints():
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
    
@app.post("/api/v1/insert")
def insertData(person:Person):
    try:
        print(person)
        print("Inserting data")
        inserted_person = gender_data.insert_one(dict(person)).inserted_id
        counter = counter_db.find_one()
        print(counter['counter'])
        if(counter['counter'] >= 5):
            run_model()
            counter_db.update_one({'name': 'counter'}, {'$set':{'counter':0}})
        else:
            update_counter()
        return {
            "insertedPerson": person,
            "success": True
        }
    except Exception as e:
        print(e)
        return {
            "Error Happened"
        }

