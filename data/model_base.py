from pydantic import BaseModel, BaseSettings
from typing import List

class bicis(BaseModel):
    season: int
    hour: int 
    workingday: int
    wheather: int
    temp: float
    atemp: float 
    hum: float
