import pandas as pd
import numpy as np


data = pd.read_csv('data/timeseries_full.csv')
#new_data

#Obteniendo los valores para la predicción 
fecha = '2011-01-01 01:00:00'

indice = data[data['new_date'] == fecha].index[0]

season = data.iloc[indice]['season']
time =  data.iloc[indice]['new_time']
workingday = data.iloc[indice]['workingday'] 
wheather = data.iloc[indice]['weathersit']
temp =  data.iloc[indice]['temp']
atemp = data.iloc[indice]['atemp']
hum = data.iloc[indice]['hum']