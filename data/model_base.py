from pydantic import BaseModel, BaseSettings
from typing import List

class bank(BaseModel):
    age: int
    amount: float 
    #workingday: int
    #wheather: int
    #temp: float
    #atemp: float 
    #hum: float
