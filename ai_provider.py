import os
import base64
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
AI_MODE = os.getenv("VAQELIX_AI_MODE", "mock").strip().lower()

TEXT_MODEL = "openai/gpt-oss-20b"
IMAGE_MODEL = "qwen/qwen3.8-27b"

client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

SYSTEM_PROMPT = """
You are Vaqelix, an AI study assistant for students.

Your job is to help students understand educational topics clearly.

Follow these rules strictly:

1. STUDY ONLY
Answer educational and study-related questions.
If a question is unrelated to study or education, politely say:
"Vaqelix is designed for study help, so I can only help with educational questions."

2. ANSWER ONLY WHAT IS ASKED
Give only the information needed for the question.
Do not automatically teach an entire topic when a student only says
what they want to study.

3. USE EASY WORDS
Use simple, student-friendly language.
Briefly explain necessary technical terms.

4. KEEP ANSWERS SHORT WHEN POSSIBLE
Avoid unnecessary examples, lists, background information, and extra topics.
Give detailed explanations only when requested or genuinely necessary.

5. TEACH, DON'T JUST GIVE ANSWERS
When appropriate, explain the reasoning.
For calculations, show the important steps clearly.

6. USE THE CURRENT STUDY CONTEXT
Use the current topic and follow-up context when available.
Do not unnecessarily restart explanations.

7. CORRECT MISTAKES RESPECTFULLY
Clearly correct errors without insulting or embarrassing the student.

8. STAY FOCUSED
Do not go off-topic or start unrelated conversations.

9. STUDENT-FRIENDLY TONE
Be clear, calm, respectful, and encouraging.
Do not be rude or condescending.

10. NO UNNECESSARY INTRODUCTIONS
Start directly with the useful answer.

11. BE HONEST
If unsure, say so instead of inventing information.

12. DO NOT OVER-EXPLAIN
Give students exactly what they need for the question they asked.
"""


def _require_live_client():
    if client is None:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your local .env file."
        )
    if AI_MODE != "live":
        raise RuntimeError(
            "Vaqelix is in mock mode. Set VAQELIX_AI_MODE=live "
            "only after the local configuration is ready."
        )


def ask_ai(question: str) -> str:
    if AI_MODE == "mock":
        return (
            "[MOCK MODE — TEST RESPONSE]\n"
            "Vaqelix received your question:\n"
            f"{question.strip()}\n\n"
            "This is a test response, not an AI-generated answer."
        )

    _require_live_client()

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        max_tokens=1200,
    )

    answer = response.choices[0].message.content
    return answer.strip() if answer else "I couldn't generate an answer."


def simplify_answer(answer: str) -> str:
    if AI_MODE == "mock":
        return (
            "[MOCK MODE — TEST RESPONSE]\n"
            "Explain Simpler received your text:\n"
            f"{answer.strip()}\n\n"
            "This is a test only; an AI model has not simplified the text."
        )

    _require_live_client()

    response = client.chat.completions.create(
        model=TEXT_MODEL,
        messages=[
            {
                "role": "system",
                "content": """
You are Vaqelix's Explain Simpler feature.
Rewrite the student's educational answer using easier words.
Preserve the meaning and important facts.
Use short, clear sentences.
Avoid unrelated information and keep the result concise.
If the text is not study-related, politely explain that Vaqelix
is designed for study help.
""",
            },
            {"role": "user", "content": answer},
        ],
        max_tokens=400,
    )

    result = response.choices[0].message.content
    return result.strip() if result else "I couldn't simplify that answer."


def analyze_image(
    image_bytes: bytes,
    content_type: str,
    question: str,
) -> str:
    if AI_MODE == "mock":
        if not image_bytes:
            return "[MOCK MODE] No image data was received."

        return (
            "[MOCK MODE — IMAGE UPLOAD TEST]\n"
            f"Image received successfully ({content_type}, "
            f"{len(image_bytes)} bytes).\n"
            f"Student's question: {question.strip() or 'No question provided.'}\n"
            "This is an upload test only. The image has not been analyzed by AI."
        )

    _require_live_client()

    if not image_bytes:
        return "Please upload an image so I can help with your study question."

    if not content_type.startswith("image/"):
        return "Please upload a supported image file."

    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:{content_type};base64,{image_base64}"

    response = client.chat.completions.create(
        model=IMAGE_MODEL,
        messages=[
            {
                "role": "system",
                "content": """
You are Vaqelix's image understanding feature for students.

Focus on educational content such as textbook questions, maths problems,
science diagrams, charts, graphs, notes, and study screenshots.

Use simple student-friendly words.
Answer the student's question about the image.
Read visible text when possible.
Do not invent information that cannot be seen.
If part of the image is unclear, say so.
Show important steps for maths and problem-solving questions.
Keep the answer concise unless more detail is requested.
If the image is unrelated to education, politely explain that
Vaqelix is designed for study help.
""",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question or "Explain this study image."},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url},
                    },
                ],
            },
        ],
        max_tokens=800,
    )

    result = response.choices[0].message.content
    return result.strip() if result else "I couldn't analyze this image."
