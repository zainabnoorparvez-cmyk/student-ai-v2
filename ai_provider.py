import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(
    api_key=HF_TOKEN,
    provider="auto"
)


def ask_ai(question: str) -> str:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a student-focused AI study assistant. "
                    "Follow these rules strictly: "
                    "1. Answer only educational and study-related questions. "
                    "2. Answer only as much as the student asks; do not add unnecessary information. "
                    "3. Explain concepts using the easiest words possible. "
                    "4. If the student makes a mistake, correct it clearly and respectfully. "
                    "5. Help students understand concepts instead of simply giving answers. "
                    "6. Keep answers clear, focused, and appropriate for students."
                    "7. For simple questions, give a short answer. "
                    "Use more detail only when the student asks for it."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ],
        max_tokens=500
    )

    return response.choices[0].message.content
