from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ai_provider import ask_ai, simplify_answer, analyze_image
from database import SessionLocal, StudySession, Message, StudyMaterial, PDFProcessingJob
import json
import re
from datetime import datetime
from pypdf import PdfReader
from io import BytesIO
from pdf2image import convert_from_bytes
import pytesseract

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500", "http://localhost:5500"],
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
    session_id: int | None = None


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
        # No session selected: answer without saved session context.
        if data.session_id is None:
            return {"answer": ask_ai(data.question)}

        # Find the selected study session.
        session = (
            db.query(StudySession)
            .filter(StudySession.id == data.session_id)
            .first()
        )

        if session is None:
            raise HTTPException(
                status_code=404,
                detail="Study session not found"
            )

        # Retrieve recent conversation messages.
        messages = (
            db.query(Message)
            .filter(Message.session_id == data.session_id)
            .order_by(Message.created_at.desc())
            .limit(10)
            .all()
        )
        messages = list(reversed(messages))

        # Retrieve saved materials belonging to this session only.
        materials = (
            db.query(StudyMaterial)
            .filter(StudyMaterial.session_id == data.session_id)
            .order_by(StudyMaterial.created_at.asc())
            .all()
        )

        material_sections = []
        remaining_chars = 16000

        for material in materials:
            content = (material.content or "").strip()

            if not content or remaining_chars <= 0:
                continue

            # Ignore placeholder image-upload tests.
            if content.startswith("[MOCK MODE — IMAGE UPLOAD TEST]"):
                continue

            section = (
                f"Material title: {material.title}\n"
                f"Material content:\n{content}"
            )
            section = section[:remaining_chars]
            material_sections.append(section)
            remaining_chars -= len(section)

        context = (
            "You are Vaqelix, a study assistant. "
            "Answer educational questions in simple, clear words. "
            "Use the saved study materials below when they are relevant. "
            "If the student asks about a specific document, check its title "
            "and content before answering. Do not claim you cannot access "
            "saved files when their text is provided below. "
            "Do not invent facts absent from the materials; be clear when "
            "the requested information is not available.\n\n"
            f"Subject: {session.subject}\n"
            f"Topic: {session.topic}\n"
        )

        if material_sections:
            context += (
                "\nSaved study materials for this session:\n"
                + "\n\n---\n\n".join(material_sections)
                + "\n"
            )

        if messages:
            context += "\nRecent conversation:\n"
            for message in messages:
                context += f"{message.role}: {message.content}\n"

        context += f"\nStudent's new question:\n{data.question}"

        answer = ask_ai(context)

        # Save the new conversation messages.
        db.add(Message(
            session_id=data.session_id,
            role="user",
            content=data.question
        ))
        db.add(Message(
            session_id=data.session_id,
            role="assistant",
            content=answer
        ))

        session.updated_at = datetime.utcnow()
        db.commit()

        return {"answer": answer}

    finally:
        db.close()


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
    # Validate the request before calling the AI.
    if data.question_count < 1:
        raise HTTPException(
            status_code=400,
            detail="Question count must be at least 1."
        )

    if data.question_count > 50:
        raise HTTPException(
            status_code=400,
            detail="Question count cannot be greater than 50."
        )

    topic = data.topic.strip()
    if not topic:
        raise HTTPException(
            status_code=400,
            detail="Please enter a quiz topic."
        )

    if data.source not in {"topic", "material"}:
        raise HTTPException(
            status_code=400,
            detail="Quiz source must be 'topic' or 'material'."
        )

    # Build the prompt from the correct source.
    if data.source == "material":
        if data.session_id is None:
            raise HTTPException(
                status_code=400,
                detail="Select a study session before creating a quiz from saved material."
            )

        db = SessionLocal()
        try:
            session = (
                db.query(StudySession)
                .filter(StudySession.id == data.session_id)
                .first()
            )

            if session is None:
                raise HTTPException(
                    status_code=404,
                    detail="Study session not found."
                )

            session_subject = session.subject
            session_topic = session.topic

            materials = (
                db.query(StudyMaterial)
                .filter(StudyMaterial.session_id == data.session_id)
                .order_by(StudyMaterial.created_at.asc())
                .all()
            )

            usable_materials = [
                material for material in materials
                if material.content and material.content.strip()
            ]

            if not usable_materials:
                raise HTTPException(
                    status_code=400,
                    detail="No readable study material was found in this session. Upload a PDF, add text, or upload an image first."
                )

            material_sections = []
            remaining_chars = 16000

            for material in usable_materials:
                if remaining_chars <= 0:
                    break

                section = (
                    f"Material title: {material.title}\\n"
                    f"Material content:\\n{material.content.strip()}"
                )
                section = section[:remaining_chars]
                material_sections.append(section)
                remaining_chars -= len(section)

            material_text = "\\n\\n---\\n\\n".join(material_sections).strip()

        finally:
            db.close()

        prompt = f"""
You are creating a study quiz for Vaqelix.

Create exactly {data.question_count} multiple-choice questions using ONLY the
saved study material provided below.

Study session subject: {session_subject}
Study session topic: {session_topic}
Student's requested focus: {topic}

IMPORTANT SOURCE RULES:
- Every question, correct answer, and explanation must be supported by the supplied material.
- Do not use general knowledge to fill gaps.
- Do not invent details that are absent from the material.
- Prefer specific facts, headings, steps, tools, checklists, and concepts actually present in the material.
- If the requested focus is not covered by the material, make questions about the material that is available rather than inventing facts.
- Avoid generic questions about what a planner is unless that is specifically explained in the material.
- Keep language simple and suitable for students.
- Each question must have exactly four options labelled A, B, C, and D.
- Exactly one option must be correct.
- The answer field must be only A, B, C, or D.
- Give a short explanation grounded in the material.
- Return a JSON array containing exactly {data.question_count} objects.
- Return ONLY valid JSON: no Markdown, no code fences, and no extra commentary.

SAVED STUDY MATERIAL:
{material_text}

Required JSON format:
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
    "explanation": "Short explanation supported by the saved material."
  }}
]
"""
    else:
        prompt = f"""
You are Vaqelix, a study assistant.

Create exactly {data.question_count} multiple-choice questions for a student
studying this topic:

{topic}

Rules:
- Keep every question relevant to the requested topic.
- Use simple student-friendly language.
- Each question must have exactly four options labelled A, B, C, and D.
- Exactly one option must be correct.
- The answer field must be only A, B, C, or D.
- Give a short, clear explanation of the correct answer.
- Return a JSON array containing exactly {data.question_count} objects.
- Return ONLY valid JSON: no Markdown, no code fences, and no extra commentary.

Required JSON format:
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

    # Do not silently return unrelated sample questions when live AI is unavailable.
    if response.startswith("[MOCK MODE — TEST RESPONSE]"):
        raise HTTPException(
            status_code=503,
            detail="Vaqelix is in mock mode, so it cannot create a real quiz. Check the local AI configuration."
        )

    # Remove Markdown fences if the model accidentally includes them.
    cleaned = response.strip()
    cleaned = re.sub(r"^```json\\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^```\\s*", "", cleaned)
    cleaned = re.sub(r"\\s*```$", "", cleaned)

    try:
        quiz = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(
            status_code=502,
            detail="The AI returned invalid quiz data. Please try again."
        )

    if not isinstance(quiz, list):
        raise HTTPException(
            status_code=502,
            detail="The AI quiz response was not a JSON list. Please try again."
        )

    validated_quiz = []
    required_options = ("A", "B", "C", "D")

    for item in quiz:
        if not isinstance(item, dict):
            continue

        question = item.get("question")
        options = item.get("options")
        answer = item.get("answer")
        explanation = item.get("explanation")

        if not isinstance(question, str) or not question.strip():
            continue
        if not isinstance(options, dict):
            continue
        if not all(
            isinstance(options.get(key), str) and options[key].strip()
            for key in required_options
        ):
            continue
        if not isinstance(answer, str) or answer.strip().upper() not in required_options:
            continue
        if not isinstance(explanation, str) or not explanation.strip():
            continue

        validated_quiz.append({
            "question": question.strip(),
            "options": {
                key: options[key].strip()
                for key in required_options
            },
            "answer": answer.strip().upper(),
            "explanation": explanation.strip()
        })

    if len(validated_quiz) < data.question_count:
        raise HTTPException(
            status_code=502,
            detail=(
                f"The AI returned only {len(validated_quiz)} valid questions "
                f"out of {data.question_count}. Please try again."
            )
        )

    return {"quiz": validated_quiz[:data.question_count]}


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
# BACKGROUND PDF PROCESSING
# =========================

from pathlib import Path
import uuid

PDF_UPLOAD_DIR = Path("uploads/pdfs")
PDF_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def process_pdf_job(job_id: int):
    db = SessionLocal()

    try:
        job = (
            db.query(PDFProcessingJob)
            .filter(PDFProcessingJob.id == job_id)
            .first()
        )

        if job is None:
            return

        pdf_path = Path(job.pdf_path)
        pdf_data = pdf_path.read_bytes()
        reader = PdfReader(BytesIO(pdf_data))

        if reader.is_encrypted:
            raise ValueError("This PDF is password-protected.")

        total_pages = len(reader.pages)

        if total_pages == 0:
            raise ValueError("This PDF contains no pages.")

        material = None

        if job.material_id:
            material = (
                db.query(StudyMaterial)
                .filter(StudyMaterial.id == job.material_id)
                .first()
            )

        if material is None:
            material = StudyMaterial(
                session_id=job.session_id,
                title=job.title,
                content=""
            )
            db.add(material)
            db.commit()
            db.refresh(material)
            job.material_id = material.id

        job.status = "processing"
        job.stage = "Extracting selectable text"
        job.total_pages = total_pages
        job.processed_pages = 0
        job.error = ""
        db.commit()

        # First, extract selectable text in small batches.
        text_pages = []
        batch_size = 5

        for start in range(0, total_pages, batch_size):
            end = min(start + batch_size, total_pages)

            for page_index in range(start, end):
                page_text = reader.pages[page_index].extract_text() or ""
                text_pages.append(page_text)

            material.content = "\n\n".join(
                text for text in text_pages if text.strip()
            ).strip()

            job.processed_pages = end
            job.stage = "Extracting selectable text"
            job.extraction_method = "text"
            db.commit()

        original_content = material.content or ""

        # If almost no text was found, OCR the pages in small batches.
        if len(original_content) < 50:
            job.stage = "Reading scanned pages (OCR)"
            job.extraction_method = "OCR"
            job.processed_pages = 0
            material.content = ""
            db.commit()

            ocr_pages = []

            for start in range(1, total_pages + 1, batch_size):
                end = min(start + batch_size - 1, total_pages)

                images = convert_from_bytes(
                    pdf_data,
                    dpi=120,
                    first_page=start,
                    last_page=end,
                    fmt="png",
                    thread_count=1
                )

                for image in images:
                    page_text = pytesseract.image_to_string(
                        image,
                        lang="eng"
                    )
                    ocr_pages.append(page_text)

                ocr_content = "\n\n".join(
                    text for text in ocr_pages if text.strip()
                ).strip()

                material.content = ocr_content
                job.processed_pages = end
                job.stage = f"Reading scanned pages (OCR): {end}/{total_pages}"
                db.commit()

            # Keep the original text if OCR did not improve it.
            if len(material.content or "") <= len(original_content):
                material.content = original_content
                job.extraction_method = "text"

        if not (material.content or "").strip():
            raise ValueError(
                "No readable text was found. The PDF may be blank or "
                "require a different OCR language."
            )

        session = (
            db.query(StudySession)
            .filter(StudySession.id == job.session_id)
            .first()
        )

        if session:
            session.updated_at = datetime.utcnow()

        job.status = "completed"
        job.stage = "Completed"
        job.processed_pages = total_pages
        job.error = ""
        db.commit()

    except Exception as exc:
        db.rollback()

        try:
            job = (
                db.query(PDFProcessingJob)
                .filter(PDFProcessingJob.id == job_id)
                .first()
            )

            if job:
                job.status = "failed"
                job.stage = "Processing failed"
                job.error = str(exc)[:4000]
                db.commit()
        except Exception:
            db.rollback()

    finally:
        db.close()


# =========================
# UPLOAD PDF STUDY MATERIAL
# =========================

@app.post("/sessions/{session_id}/materials/upload-pdf")
async def upload_pdf(
    session_id: int,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    db = SessionLocal()
    saved_path = None

    try:
        session = (
            db.query(StudySession)
            .filter(StudySession.id == session_id)
            .first()
        )

        if session is None:
            raise HTTPException(
                status_code=404,
                detail="Study session not found"
            )

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No file selected"
            )

        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Only PDF files are allowed"
            )

        pdf_data = await file.read()

        if not pdf_data:
            raise HTTPException(
                status_code=400,
                detail="The PDF file is empty"
            )

        try:
            reader = PdfReader(BytesIO(pdf_data))

            if reader.is_encrypted:
                raise HTTPException(
                    status_code=400,
                    detail="This PDF is password-protected. Please upload an unlocked PDF."
                )

            page_count = len(reader.pages)

            if page_count == 0:
                raise HTTPException(
                    status_code=400,
                    detail="This PDF contains no pages."
                )

        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Could not read this PDF: {exc}"
            ) from exc

        safe_name = f"{uuid.uuid4().hex}.pdf"
        saved_path = PDF_UPLOAD_DIR / safe_name
        saved_path.write_bytes(pdf_data)

        job = PDFProcessingJob(
            session_id=session_id,
            title=file.filename[:200],
            pdf_path=str(saved_path),
            status="queued",
            stage="Waiting to start",
            total_pages=page_count,
            processed_pages=0
        )

        db.add(job)
        db.commit()
        db.refresh(job)

        background_tasks.add_task(process_pdf_job, job.id)

        return {
            "message": "PDF accepted for background processing",
            "job_id": job.id,
            "session_id": session_id,
            "title": job.title,
            "pages": page_count,
            "status": job.status
        }

    except HTTPException:
        if saved_path and saved_path.exists():
            saved_path.unlink()
        raise

    except Exception as exc:
        db.rollback()

        if saved_path and saved_path.exists():
            saved_path.unlink()

        raise HTTPException(
            status_code=500,
            detail=f"Could not start PDF processing: {exc}"
        ) from exc

    finally:
        await file.close()
        db.close()


# =========================
# CHECK PDF PROCESSING STATUS
# =========================

@app.get("/pdf-jobs/{job_id}")
def get_pdf_job_status(job_id: int):
    db = SessionLocal()

    try:
        job = (
            db.query(PDFProcessingJob)
            .filter(PDFProcessingJob.id == job_id)
            .first()
        )

        if job is None:
            raise HTTPException(
                status_code=404,
                detail="PDF processing job not found"
            )

        progress = 0

        if job.total_pages > 0:
            progress = round(
                job.processed_pages / job.total_pages * 100
            )

        if job.status == "completed":
            progress = 100

        return {
            "job_id": job.id,
            "session_id": job.session_id,
            "title": job.title,
            "status": job.status,
            "stage": job.stage,
            "total_pages": job.total_pages,
            "processed_pages": job.processed_pages,
            "progress": progress,
            "material_id": job.material_id,
            "extraction_method": job.extraction_method,
            "error": job.error
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

