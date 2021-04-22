import pandas as pd
import numpy as np


data = pd.read_csv('data/banksim_adj.csv')

#fecha = '2011-01-01 01:00:00'

#indice = data[data['new_date'] == fecha].index[0]

age = data['age']
amount =  data['amount']
M: int
es_barsandrestaurants: int
es_contents: int
es_fashion: int 
es_food: int
es_health: int 
es_home: int 
es_hotelservices: int 
es_hyper: int
es_leisure: int 
es_otherservices: int 
es_sportsandtoys: int 
es_tech: int 
es_transportation: int 
es_travel: int 
#workingday = data.iloc[indice]['workingday'] 
#wheather = data.iloc[indice]['weathersit']
#temp =  data.iloc[indice]['temp']
#atemp = data.iloc[indice]['atemp']
#hum = data.iloc[indice]['hum']