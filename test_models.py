import pandas as pd
import numpy as np
import pickle

from pydantic import BaseModel, BaseSettings
from data.model_base import bank
from typing import List


rf_pickle = open('models/RFregression.pkl', 'rb')
rf_model = pickle.load(rf_pickle)

age = 3
amount = 18.76
#workingday = 0.0
#wheather = 1.0
#temp = 0.22
#atemp = 0.2727
#hum = 0.8

prediction = rf_model.predict([[age, amount]])
print('The number of bikes is:', prediction)

origin = {
  "age": 3,
  "amount": 18.76,
  #"workingday": 0,
  #"wheather": 1,
  #"temp": 0.22,
  #"atemp": 0.2727,
  #"hum": 0.8
}

tran_ = list(origin.values())
data_in = np.array(tran_).reshape(1,7)
rf_model.predict(data_in)

def predict_demand(bikes:bank):
    data = bikes
    age = data['age']
    amount =  data['amount']
    #season = df['season']
    #hour = df['hour']
    #workingday = df['workingday']
    #wheather = df['wheather']
    #temp = df['temp']
    #atemp = df['atemp']
    #hum = df['hum']


    return data

valores = [1, 6, 0, 1, 0.22, 0.2727, 0.8]
pre = predict_demand(origin)

class bank(BaseSettings):
    age = int 
    amount = float
    #season: int
    #hour: int 
    #workingday: int
    #wheather: int
    #temp: float
    #atemp: float 
    #hum: float

def predict_val(valores:list):
    rf_pickle = open('models/RFregression.pkl', 'rb')
    rf_model = pickle.load(rf_pickle)
    prediction = rf_model.predict(valores)
    return prediction

predict_val([[1,1,0,1,0.22,0.27,0.8]])[0]