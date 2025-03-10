from pydantic import BaseModel

class Student(BaseModel):
    name: str
    age: int
    city: str


new_student = {
    "name": "John Doe",
    "age": 20,
    "city": "New York"
}

student = Student(**new_student)

print(student)





