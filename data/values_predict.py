import pandas as pd
import numpy as np


data = pd.read_csv('data/banksim_adj.csv')

#fecha = '2011-01-01 01:00:00'

#indice = data[data['new_date'] == fecha].index[0]

age = data['age']
amount =  data['amount']
#workingday = data.iloc[indice]['workingday'] 
#wheather = data.iloc[indice]['weathersit']
#temp =  data.iloc[indice]['temp']
#atemp = data.iloc[indice]['atemp']
#hum = data.iloc[indice]['hum']