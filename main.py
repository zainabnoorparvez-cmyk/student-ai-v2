from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ai_provider import ask_ai, simplify_answer
from database import SessionLocal, StudySession, Message
import json
import re
from datetime import datetime

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Question(BaseModel):
    question: str
    session_id: int | None = None


class Answer(BaseModel):
    answer: str


class PracticeRequest(BaseModel):
    topic: str


class QuizRequest(BaseModel):
    topic: str
    question_count: int = 5
    source: str = "topic"


class CreateSessionRequest(BaseModel):
    subject: str
    topic: str


class SaveMessageRequest(BaseModel):
    role: str
    content: str


@app.get("/")
def home():
    return {"message": "Vaqelix is running!"}


# -------------------------
# ASK VAQELIX
# -------------------------

@app.post("/ask")
def ask_question(data: Question):

    if data.session_id is None:
        return {
            "question": data.question,
            "answer": ask_ai(data.question)
        }

    db = SessionLocal()

    try:
        study_session = (
            db.query(StudySession)
            .filter(StudySession.id == data.session_id)
            .first()
        )

        if study_session is None:
            raise HTTPException(
                status_code=404,
                detail="Study session not found"
            )

        messages = (
            db.query(Message)
            .filter(Message.session_id == data.session_id)
            .order_by(Message.created_at.asc())
            .all()
        )

        context_parts = [
            f"Subject: {study_session.subject}",
            f"Topic: {study_session.topic}"
        ]

        if messages:
            context_parts.append("\nPrevious study conversation:")

            for message in messages[-10:]:
                role_name = (
                    "Student"
                    if message.role == "user"
                    else "Vaqelix"
                )

                context_parts.append(
                    f"{role_name}: {message.content}"
                )

        context_parts.append(
            f"\nStudent's new question: {data.question}"
        )

        contextual_prompt = "\n".join(context_parts)

        answer = ask_ai(contextual_prompt)

        return {
            "question": data.question,
            "answer": answer,
            "session_id": data.session_id,
            "subject": study_session.subject,
            "topic": study_session.topic
        }

    finally:
        db.close()


# -------------------------
# EXPLAIN SIMPLER
# -------------------------

@app.post("/simplify")
def simplify_question(data: Answer):
    return {
        "answer": simplify_answer(data.answer)
    }


# -------------------------
# PRACTICE QUESTIONS
# -------------------------

@app.post("/practice")
def create_practice_questions(data: PracticeRequest):

    prompt = f"""
Create 5 practice questions about "{data.topic}" for a student.

The questions are for learning and practice, NOT a quiz.

Return ONLY valid JSON.
Do not use Markdown.
Do not add explanations.
Do not use code fences.

Use exactly this format:

[
  {{
    "question": "Practice question text",
    "answer": "Short and simple answer"
  }}
]

Rules:
- Exactly 5 questions.
- Questions should test understanding.
- Use simple student-friendly language.
- Questions should be useful for studying.
- Keep answers short and clear.
"""

    raw_practice = ask_ai(prompt)

    try:
        json_match = re.search(
            r"\[.*\]",
            raw_practice,
            re.DOTALL
        )

        if not json_match:
            raise ValueError("Practice JSON not found")

        practice = json.loads(json_match.group())

        return {
            "topic": data.topic,
            "questions": practice
        }

    except Exception:
        return {
            "topic": data.topic,
            "questions": [],
            "error": (
                "Vaqelix could not create the practice questions "
                "correctly. Please try again."
            )
        }


# -------------------------
# QUIZ
# -------------------------

@app.post("/quiz")
def create_quiz(data: QuizRequest):

    question_count = data.question_count

    if question_count < 1:
        question_count = 1

    if question_count > 50:
        question_count = 50

    if data.source == "material":

        prompt = f"""
Create exactly {question_count} multiple-choice quiz questions
ONLY from the study material provided below.

Do not use information that is not present in the study material.

Study Material:
{data.topic}

Return ONLY valid JSON.
Do not use Markdown.
Do not add explanations.
Do not use code fences.

Use exactly this format:

[
  {{
    "question": "Question text",
    "options": {{
      "A": "Option A",
      "B": "Option B",
      "C": "Option C",
      "D": "Option D"
    }},
    "correct": "A"
  }}
]

Rules:
- Exactly {question_count} questions.
- Every question must have exactly 4 options.
- The correct field must contain only A, B, C, or D.
- Every question must be answerable from the provided study material.
- Do not introduce outside facts.
- Questions should test understanding, not just memorization.
- Use simple student-friendly language.
"""

    else:

        prompt = f"""
Create exactly {question_count} multiple-choice quiz questions
about the topic "{data.topic}".

The quiz should help a student test their understanding of the topic.

IMPORTANT:
Cover the major and important concepts of the topic.
Do not create random questions about only one small part.
Distribute the questions across different important concepts.
Avoid repeating the same concept.

Return ONLY valid JSON.
Do not use Markdown.
Do not add explanations.
Do not use code fences.

Use exactly this format:

[
  {{
    "question": "Question text",
    "options": {{
      "A": "Option A",
      "B": "Option B",
      "C": "Option C",
      "D": "Option D"
    }},
    "correct": "A"
  }}
]

Rules:
- Exactly {question_count} questions.
- Every question must have exactly 4 options.
- The correct field must contain only A, B, C, or D.
- Questions should test understanding.
- Cover major concepts.
- Avoid duplicate or nearly identical questions.
- Use simple student-friendly language.
"""

    raw_quiz = ask_ai(prompt)

    try:

        json_match = re.search(
            r"\[.*\]",
            raw_quiz,
            re.DOTALL
        )

        if not json_match:
            raise ValueError("Quiz JSON not found")

        quiz = json.loads(json_match.group())

        if not isinstance(quiz, list):
            raise ValueError("Quiz is not a list")

        cleaned_quiz = []

        for question in quiz:

            if not isinstance(question, dict):
                continue

            if (
                "question" not in question
                or "options" not in question
                or "correct" not in question
            ):
                continue

            options = question["options"]

            if not isinstance(options, dict):
                continue

            required_options = ["A", "B", "C", "D"]

            if not all(
                option in options
                for option in required_options
            ):
                continue

            if question["correct"] not in required_options:
                continue

            cleaned_quiz.append({
                "question": question["question"],
                "options": {
                    "A": options["A"],
                    "B": options["B"],
                    "C": options["C"],
                    "D": options["D"]
                },
                "correct": question["correct"]
            })

        if len(cleaned_quiz) < question_count:

            return {
                "topic": data.topic,
                "quiz": cleaned_quiz,
                "error": (
                    "Vaqelix could not create the requested number "
                    "of valid questions. Please try again."
                )
            }

        return {
            "topic": data.topic,
            "quiz": cleaned_quiz[:question_count]
        }

    except Exception:

        return {
            "topic": data.topic,
            "quiz": [],
            "error": (
                "Vaqelix could not create the quiz correctly. "
                "Please try again."
            )
        }


# -------------------------
# CREATE STUDY SESSION
# -------------------------

@app.post("/sessions")
def create_session(data: CreateSessionRequest):

    db = SessionLocal()

    try:

        study_session = StudySession(
            subject=data.subject,
            topic=data.topic,
            status="Learning",
            summary=""
        )

        db.add(study_session)
        db.commit()
        db.refresh(study_session)

        return {
            "id": study_session.id,
            "subject": study_session.subject,
            "topic": study_session.topic,
            "status": study_session.status,
            "summary": study_session.summary,
            "created_at": study_session.created_at,
            "updated_at": study_session.updated_at
        }

    finally:
        db.close()


# -------------------------
# GET ALL SESSIONS
# -------------------------

@app.get("/sessions")
def get_sessions():

    db = SessionLocal()

    try:

        sessions = (
            db.query(StudySession)
            .order_by(StudySession.updated_at.desc())
            .all()
        )

        return [
            {
                "id": session.id,
                "subject": session.subject,
                "topic": session.topic,
                "status": session.status,
                "summary": session.summary,
                "created_at": session.created_at,
                "updated_at": session.updated_at
            }
            for session in sessions
        ]

    finally:
        db.close()


# -------------------------
# GET ONE SESSION
# -------------------------

@app.get("/sessions/{session_id}")
def get_session(session_id: int):

    db = SessionLocal()

    try:

        study_session = (
            db.query(StudySession)
            .filter(StudySession.id == session_id)
            .first()
        )

        if study_session is None:
            raise HTTPException(
                status_code=404,
                detail="Study session not found"
            )

        messages = (
            db.query(Message)
            .filter(Message.session_id == session_id)
            .order_by(Message.created_at.asc())
            .all()
        )

        return {
            "id": study_session.id,
            "subject": study_session.subject,
            "topic": study_session.topic,
            "status": study_session.status,
            "summary": study_session.summary,
            "created_at": study_session.created_at,
            "updated_at": study_session.updated_at,
            "messages": [
                {
                    "id": message.id,
                    "role": message.role,
                    "content": message.content,
                    "created_at": message.created_at
                }
                for message in messages
            ]
        }

    finally:
        db.close()


# -------------------------
# SAVE MESSAGE
# -------------------------

@app.post("/sessions/{session_id}/messages")
def save_message(
    session_id: int,
    data: SaveMessageRequest
):

    if data.role not in ["user", "assistant"]:

        raise HTTPException(
            status_code=400,
            detail="Role must be user or assistant"
        )

    db = SessionLocal()

    try:

        study_session = (
            db.query(StudySession)
            .filter(StudySession.id == session_id)
            .first()
        )

        if study_session is None:

            raise HTTPException(
                status_code=404,
                detail="Study session not found"
            )

        message = Message(
            session_id=session_id,
            role=data.role,
            content=data.content
        )

        db.add(message)

        study_session.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(message)

        return {
            "id": message.id,
            "session_id": message.session_id,
            "role": message.role,
            "content": message.content,
            "created_at": message.created_at
        }

    finally:
        db.close()


# -------------------------
# DELETE SESSION
# -------------------------

@app.delete("/sessions/{session_id}")
def delete_session(session_id: int):

    db = SessionLocal()

    try:

        study_session = (
            db.query(StudySession)
            .filter(StudySession.id == session_id)
            .first()
        )

        if study_session is None:

            raise HTTPException(
                status_code=404,
                detail="Study session not found"
            )

        db.delete(study_session)
        db.commit()

        return {
            "message": "Study session deleted successfully",
            "id": session_id
        }

    finally:
        db.close()
