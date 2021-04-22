from pydantic import BaseModel, BaseSettings
from typing import List

class bank(BaseModel):
    age: int
    amount: float 
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