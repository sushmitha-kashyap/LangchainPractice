from pydantic import BaseModel,EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name: str = 'defaultName'
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0,lt=10,default=5)

new_student = {'age' : 30, 'email':'abc', 'cgpa':11}

student = Student(**new_student)

print(student)