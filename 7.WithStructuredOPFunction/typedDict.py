
from typing import TypedDict

class Person(TypedDict):
    name: str
    age:int

new_Person: Person = {'name': 'Sushmitha' , 'age': 26}

print(new_Person)
