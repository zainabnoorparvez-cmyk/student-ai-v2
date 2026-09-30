from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ai_provider import ask_ai, simplify_answer, analyze_image
from database import SessionLocal, StudySession, Message, StudyMaterial
import json
import re
from datetime import datetime
from pypdf import PdfReader
from io import BytesIO

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# REQUEST MODELS
# =========================

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


class StudyMaterialRequest(BaseModel):
    title: str
    content: str


# =========================
# HOME
# =========================

app.frontend("/", directory="frontend")

# =========================
# ASK VAQELIX
# =========================

@app.post("/ask")
def ask_question(data: Question):
    db = SessionLocal()

    try:
        # No session selected
        if data.session_id is None:
            answer = ask_ai(data.question)
            return {"answer": answer}

        # Find current study session
        session = db.query(StudySession).filter(
            StudySession.id == data.session_id
        ).first()

        if not session:
            raise HTTPException(
                status_code=404,
                detail="Study session not found"
            )

        # Get recent messages
        messages = (
            db.query(Message)
            .filter(Message.session_id == data.session_id)
            .order_by(Message.created_at.desc())
            .limit(10)
            .all()
        )

        messages = list(reversed(messages))

        context = f"""
The student is currently studying:

Subject: {session.subject}
Topic: {session.topic}

Use this study context when answering the student's question.
Do not unnecessarily restart the whole topic.
"""

        if messages:
            context += "\nRecent conversation:\n"

            for message in messages:
                context += f"{message.role}: {message.content}\n"

        context += f"\nStudent's new question:\n{data.question}"

        answer = ask_ai(context)

        # Save student message
        student_message = Message(
            session_id=data.session_id,
            role="user",
            content=data.question
        )

        # Save AI response
        assistant_message = Message(
            session_id=data.session_id,
            role="assistant",
            content=answer
        )

        db.add(student_message)
        db.add(assistant_message)

        # Update session activity
        session.updated_at = datetime.utcnow()

        db.commit()

        return {"answer": answer}

    finally:
        db.close()


# =========================
# EXPLAIN SIMPLER
# =========================

@app.post("/simplify")
def simplify(data: Answer):
    answer = simplify_answer(data.answer)

    return {
        "answer": answer
    }


# =========================
# PRACTICE QUESTIONS
# =========================

@app.post("/practice")
def practice_questions(data: PracticeRequest):

    prompt = f"""
Create short practice questions for a student studying this topic:

{data.topic}

Rules:
- Create 5 short questions.
- Keep them educational.
- Use simple language.
- Questions should help the student understand and remember the topic.
- Do not provide answers.
- Return ONLY valid JSON.

Use this exact format:

[
  "Question 1",
  "Question 2",
  "Question 3",
  "Question 4",
  "Question 5"
]
"""

    response = ask_ai(prompt)

    try:
        questions = json.loads(response)

        if not isinstance(questions, list):
            raise ValueError("Invalid format")

        return {
            "questions": questions
        }

    except Exception:
        return {
            "questions": [
                response
            ]
        }


# =========================
# QUIZ
# =========================

@app.post("/quiz")
def create_quiz(data: QuizRequest):

    if data.question_count < 1:
        raise HTTPException(
            status_code=400,
            detail="Question count must be at least 1"
        )

    if data.question_count > 50:
        raise HTTPException(
            status_code=400,
            detail="Question count cannot be greater than 50"
        )

    # =========================
    # QUIZ FROM SAVED MATERIAL
    # =========================

    if data.source == "material":

        prompt = f"""
Create exactly {data.question_count} multiple-choice questions
ONLY from the following study material.

Do not use information outside the material.

STUDY MATERIAL:

{data.topic}

Rules:
- Every question must be answerable from the material.
- Use simple student-friendly language.
- Each question must have exactly 4 options.
- Options must be labeled A, B, C, and D.
- Only one option can be correct.
- The correct answer must be one of A, B, C, or D.
- Give a short and clear explanation of why the correct answer is correct.
- The explanation must help a student understand the concept.
- Return ONLY valid JSON.
- Do not add Markdown or code fences.

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
    "answer": "A",
    "explanation": "Short explanation of the correct answer."
  }}
]
"""

    # =========================
    # QUIZ FROM TOPIC
    # =========================

    else:

        prompt = f"""
Create exactly {data.question_count} multiple-choice questions
for a student studying this topic:

{data.topic}

Rules:
- Keep questions educational.
- Use simple student-friendly language.
- Each question must have exactly 4 options.
- Options must be labeled A, B, C, and D.
- Only one option can be correct.
- The correct answer must be one of A, B, C, or D.
- Give a short and clear explanation of why the correct answer is correct.
- The explanation must help a student understand the concept.
- Return ONLY valid JSON.
- Do not add Markdown or code fences.

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
    "answer": "A",
    "explanation": "Short explanation of the correct answer."
  }}
]
"""

    response = ask_ai(prompt)

    # Remove markdown code fences if AI adds them
    cleaned = response.strip()

    cleaned = re.sub(
        r"^```json\s*",
        "",
        cleaned,
        flags=re.IGNORECASE
    )

    cleaned = re.sub(
        r"^```\s*",
        "",
        cleaned
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned
    )

    try:
        quiz = json.loads(cleaned)

    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail="AI returned invalid quiz data."
        )

    if not isinstance(quiz, list):
        raise HTTPException(
            status_code=500,
            detail="Quiz format is invalid."
        )

    validated_quiz = []

    for item in quiz:

        if not isinstance(item, dict):
            continue

        question = item.get("question")
        options = item.get("options")
        answer = item.get("answer")
        explanation = item.get("explanation")

        if not question:
            continue

        if not isinstance(options, dict):
            continue

        required_options = ["A", "B", "C", "D"]

        if not all(option in options for option in required_options):
            continue

        if answer not in required_options:
            continue

        if not explanation:
            explanation = (
                "Review the question and the correct answer "
                "to understand this concept."
            )

        validated_quiz.append({
            "question": question,
            "options": {
                "A": options["A"],
                "B": options["B"],
                "C": options["C"],
                "D": options["D"]
            },
            "answer": answer,
            "explanation": explanation
        })

    if len(validated_quiz) < data.question_count:
        raise HTTPException(
            status_code=500,
            detail="AI did not generate enough valid quiz questions."
        )

    return {
        "quiz": validated_quiz[:data.question_count]
    }

# =========================
# CREATE STUDY SESSION
# =========================

@app.post("/sessions")
def create_session(data: CreateSessionRequest):

    db = SessionLocal()

    try:
        subject = data.subject.strip()
        topic = data.topic.strip()

        if not subject:
            raise HTTPException(
                status_code=400,
                detail="Subject cannot be empty"
            )

        if not topic:
            raise HTTPException(
                status_code=400,
                detail="Topic cannot be empty"
            )

        session = StudySession(
            subject=subject,
            topic=topic
        )

        db.add(session)
        db.commit()
        db.refresh(session)

        return {
            "id": session.id,
            "subject": session.subject,
            "topic": session.topic,
            "created_at": session.created_at,
            "updated_at": session.updated_at
        }

    finally:
        db.close()


# =========================
# GET ALL STUDY SESSIONS
# =========================

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
                "created_at": session.created_at,
                "updated_at": session.updated_at
            }
            for session in sessions
        ]

    finally:
        db.close()


# =========================
# GET ONE STUDY SESSION
# =========================

@app.get("/sessions/{session_id}")
def get_session(session_id: int):

    db = SessionLocal()

    try:
        session = db.query(StudySession).filter(
            StudySession.id == session_id
        ).first()

        if not session:
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
            "id": session.id,
            "subject": session.subject,
            "topic": session.topic,
            "created_at": session.created_at,
            "updated_at": session.updated_at,
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


# =========================
# SAVE MESSAGE
# =========================

@app.post("/sessions/{session_id}/messages")
def save_message(
    session_id: int,
    data: SaveMessageRequest
):

    db = SessionLocal()

    try:
        session = db.query(StudySession).filter(
            StudySession.id == session_id
        ).first()

        if not session:
            raise HTTPException(
                status_code=404,
                detail="Study session not found"
            )

        role = data.role.strip()
        content = data.content.strip()

        if not role:
            raise HTTPException(
                status_code=400,
                detail="Message role cannot be empty"
            )

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Message content cannot be empty"
            )

        message = Message(
            session_id=session_id,
            role=role,
            content=content
        )

        db.add(message)

        session.updated_at = datetime.utcnow()

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


# =========================
# DELETE STUDY SESSION
# =========================

@app.delete("/sessions/{session_id}")
def delete_session(session_id: int):

    db = SessionLocal()

    try:
        session = db.query(StudySession).filter(
            StudySession.id == session_id
        ).first()

        if not session:
            raise HTTPException(
                status_code=404,
                detail="Study session not found"
            )

        db.delete(session)
        db.commit()

        return {
            "message": "Study session deleted successfully"
        }

    finally:
        db.close()


# =========================
# SAVE STUDY MATERIAL
# =========================

@app.post("/sessions/{session_id}/materials")
def save_study_material(
    session_id: int,
    data: StudyMaterialRequest
):

    db = SessionLocal()

    try:
        session = db.query(StudySession).filter(
            StudySession.id == session_id
        ).first()

        if not session:
            raise HTTPException(
                status_code=404,
                detail="Study session not found"
            )

        title = data.title.strip()
        content = data.content.strip()

        if not title:
            raise HTTPException(
                status_code=400,
                detail="Material title cannot be empty"
            )

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Study material cannot be empty"
            )

        material = StudyMaterial(
            session_id=session_id,
            title=title,
            content=content
        )

        db.add(material)

        session.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(material)

        return {
            "id": material.id,
            "session_id": material.session_id,
            "title": material.title,
            "content": material.content,
            "created_at": material.created_at,
            "updated_at": material.updated_at
        }

    finally:
        db.close()


# =========================
# UPLOAD PDF STUDY MATERIAL
# =========================

@app.post("/sessions/{session_id}/materials/upload-pdf")
async def upload_pdf(
    session_id: int,
    file: UploadFile = File(...)
):

    db = SessionLocal()

    try:
        # Check study session
        session = db.query(StudySession).filter(
            StudySession.id == session_id
        ).first()

        if not session:
            raise HTTPException(
                status_code=404,
                detail="Study session not found"
            )

        # Check filename
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No file selected"
            )

        # Only PDFs
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed"
            )

        # Read PDF
        pdf_data = await file.read()

        if not pdf_data:
            raise HTTPException(
                status_code=400,
                detail="The PDF file is empty"
            )

        # Read PDF pages
        reader = PdfReader(BytesIO(pdf_data))

        extracted_pages = []

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                extracted_pages.append(page_text)

        # Combine extracted text
        content = "\n\n".join(extracted_pages).strip()

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Could not extract readable text from this PDF."
            )

        # Save extracted PDF text as study material
        material = StudyMaterial(
            session_id=session_id,
            title=file.filename,
            content=content
        )

        db.add(material)

        # Update session activity
        session.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(material)

        return {
            "message": "PDF uploaded successfully",
            "id": material.id,
            "session_id": material.session_id,
            "title": material.title,
            "pages": len(reader.pages),
            "content_length": len(content)
        }

    finally:
        db.close()


# =========================
# UPLOAD IMAGE STUDY MATERIAL
# =========================

@app.post("/sessions/{session_id}/materials/upload-image")
async def upload_image(
    session_id: int,
    file: UploadFile = File(...),
    question: str = "Explain the educational content in this image clearly."
):

    db = SessionLocal()

    try:
        # Check study session
        session = db.query(StudySession).filter(
            StudySession.id == session_id
        ).first()

        if not session:
            raise HTTPException(
                status_code=404,
                detail="Study session not found"
            )

        # Check filename
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No file selected"
            )

        # Allowed image types
        allowed_types = [
            "image/jpeg",
            "image/png",
            "image/webp"
        ]

        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail="Only JPG, PNG, and WEBP images are allowed"
            )

        # Read image
        image_data = await file.read()

        if not image_data:
            raise HTTPException(
                status_code=400,
                detail="The image file is empty"
            )

        # Ask vision model to understand the image
        answer = analyze_image(
            image_bytes=image_data,
            content_type=file.content_type,
            question=question
        )

        # Save the AI understanding as study material
        material = StudyMaterial(
            session_id=session_id,
            title=file.filename,
            content=answer
        )

        db.add(material)

        # Update session activity
        session.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(material)

        return {
            "message": "Image uploaded and analyzed successfully",
            "id": material.id,
            "session_id": material.session_id,
            "title": material.title,
            "content": material.content
        }

    finally:
        db.close()


# =========================
# GET STUDY MATERIALS
# =========================

@app.get("/sessions/{session_id}/materials")
def get_study_materials(session_id: int):

    db = SessionLocal()

    try:
        session = db.query(StudySession).filter(
            StudySession.id == session_id
        ).first()

        if not session:
            raise HTTPException(
                status_code=404,
                detail="Study session not found"
            )

        materials = (
            db.query(StudyMaterial)
            .filter(StudyMaterial.session_id == session_id)
            .order_by(StudyMaterial.created_at.desc())
            .all()
        )

        return [
            {
                "id": material.id,
                "session_id": material.session_id,
                "title": material.title,
                "content": material.content,
                "created_at": material.created_at,
                "updated_at": material.updated_at
            }
            for material in materials
        ]

    finally:
        db.close()


# =========================
# GET ONE STUDY MATERIAL
# =========================

@app.get("/materials/{material_id}")
def get_study_material(material_id: int):

    db = SessionLocal()

    try:
        material = db.query(StudyMaterial).filter(
            StudyMaterial.id == material_id
        ).first()

        if not material:
            raise HTTPException(
                status_code=404,
                detail="Study material not found"
            )

        return {
            "id": material.id,
            "session_id": material.session_id,
            "title": material.title,
            "content": material.content,
            "created_at": material.created_at,
            "updated_at": material.updated_at
        }

    finally:
        db.close()


# =========================
# UPDATE STUDY MATERIAL
# =========================

@app.put("/materials/{material_id}")
def update_study_material(
    material_id: int,
    data: StudyMaterialRequest
):

    db = SessionLocal()

    try:
        material = db.query(StudyMaterial).filter(
            StudyMaterial.id == material_id
        ).first()

        if not material:
            raise HTTPException(
                status_code=404,
                detail="Study material not found"
            )

        title = data.title.strip()
        content = data.content.strip()

        if not title:
            raise HTTPException(
                status_code=400,
                detail="Material title cannot be empty"
            )

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Study material cannot be empty"
            )

        material.title = title
        material.content = content
        material.updated_at = datetime.utcnow()

        # Update related session
        session = db.query(StudySession).filter(
            StudySession.id == material.session_id
        ).first()

        if session:
            session.updated_at = datetime.utcnow()

        db.commit()
        db.refresh(material)

        return {
            "id": material.id,
            "session_id": material.session_id,
            "title": material.title,
            "content": material.content,
            "created_at": material.created_at,
            "updated_at": material.updated_at
        }

    finally:
        db.close()


# =========================
# DELETE STUDY MATERIAL
# =========================

@app.delete("/materials/{material_id}")
def delete_study_material(material_id: int):

    db = SessionLocal()

    try:
        material = db.query(StudyMaterial).filter(
            StudyMaterial.id == material_id
        ).first()

        if not material:
            raise HTTPException(
                status_code=404,
                detail="Study material not found"
            )

        db.delete(material)
        db.commit()

        return {
            "message": "Study material deleted successfully"
        }

    finally:
        db.close()

