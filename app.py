import uvicorn
from fastapi import FastAPI
import pandas as pd
import numpy as np
import pickle

from pydantic import BaseModel, BaseSettings
from data.model_base import bank
from typing import List
from data.list_modebase import values_list

#fecha = '2011-01-01 01:00:00'
data = pd.read_csv('data/banksim_adj.csv')
app = FastAPI()

rf_pickle = open('models/RFregression.pkl', 'rb')
rf_model = pickle.load(rf_pickle)

@app.get('/get_date')
async def get_date(date_time: str):
    #indice = data[data['new_date'] == date_time].index[0]

    age = data['age']
    amount =  data['amount']
    M = data['M']
    es_barsandrestaurants = data['es_barsandrestaurants']
    es_contents = data['es_contents']
    es_fashion = data['es_fashion']
    es_food = data['es_food']
    es_health = data['es_health']
    es_home = data['es_home']
    es_hotelservices = data['es_hotelservices']
    es_hyper = data['es_hyper']
    es_leisure = data['es_leisure']
    es_otherservices = data['es_otherservices']
    es_sportsandtoys = data['es_sportsandtoys']
    es_tech = data['es_tech']
    es_transportation = data['es_transportation']
    es_travel = data['es_travel']
    #season = data.iloc[indice]['season']
    #time =  data.iloc[indice]['new_time']
    #workingday = data.iloc[indice]['workingday'] 
    #wheather = data.iloc[indice]['weathersit']
    #temp =  data.iloc[indice]['temp']
    #atemp = data.iloc[indice]['atemp']
    #hum = data.iloc[indice]['hum']

    values = {'age': age, 'amount': amount,  M :'M', es_barsandrestaurants:'es_barsandrestaurants',
                es_contents:'es_contents', es_fashion:'es_fashion', es_food:'es_food',
                es_health:'es_health', es_home :'es_home', es_hotelservices:'es_hotelservices',
                es_hyper:'es_hyper', es_leisure:'es_leisure', es_otherservices:'es_otherservices',
                es_sportsandtoys:'es_sportsandtoys', es_tech:'es_tech', es_transportation:'es_transportation',
                es_travel:'es_travel'}#, 'workday':workingday,'wheather': wheather, 'temp': temp, 'atemp': atemp, 'hum':hum}
    
    return values 

@app.post('/predict')
async def predict_demand(bikes:bank):
    rf_pickle = open('models/RFregression.pkl', 'rb')
    rf_model = pickle.load(rf_pickle)
    data = bikes.dict()
    age = data['age']
    amount =  data['amount']
    M = data['M']
    es_barsandrestaurants = data['es_barsandrestaurants']
    es_contents = data['es_contents']
    es_fashion = data['es_fashion']
    es_food = data['es_food']
    es_health = data['es_health']
    es_home = data['es_home']
    es_hotelservices = data['es_hotelservices']
    es_hyper = data['es_hyper']
    es_leisure = data['es_leisure']
    es_otherservices = data['es_otherservices']
    es_sportsandtoys = data['es_sportsandtoys']
    es_tech = data['es_tech']
    es_transportation = data['es_transportation']
    es_travel = data['es_travel']
    #season = df['season']
    #hour = df['hour']
    #workingday = df['workingday']
    #wheather = df['wheather']
    #temp = df['temp']
    #atemp = df['atemp']
    #hum = df['hum']

    get_val = list(data.values())
    
    prediction = round(rf_model.predict([get_val])[0],3)
    result = {'the person committed fraud': prediction}
    return result

if __name__ == '__main__':
    uvicorn.run(app, host = '127.0.0.1', port=8000)


