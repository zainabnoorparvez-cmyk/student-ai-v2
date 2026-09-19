from fastapi import FastAPI
from pydantic import BaseModel
from ai_provider import ask_ai

app = FastAPI()


class Question(BaseModel):
    question: str


@app.get("/")
def home():
    return {"message": "Student AI V2 is running!"}


@app.post("/ask")
def ask_question(data: Question):
    return {
        "question": data.question,
        "answer": ask_ai(data.question)
    }
