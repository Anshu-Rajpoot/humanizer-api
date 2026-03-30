from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import os, re, random, asyncio, httpx, json

# ------------------ Load API Key ------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found!")

# ------------------ FastAPI Setup ------------------
app = FastAPI(title="Pro AI Humanizer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Request Model ------------------
class HumanizeRequest(BaseModel):
    text: str

# ------------------ Load Human Reference Pool ------------------
if not os.path.exists("human_reference.json"):
    raise FileNotFoundError("Please provide human_reference.json with example human paragraphs")
with open("human_reference.json", "r", encoding="utf-8") as f:
    HUMAN_POOL = json.load(f)

# ------------------ Retrieve human examples ------------------
def retrieve_human_examples(n: int = 3):
    return random.sample(HUMAN_POOL, min(n, len(HUMAN_POOL)))

# ------------------ Chunk Text ------------------
def chunk_text(text: str, max_words: int = 350) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# ------------------ Async GPT-4o-mini Humanize ------------------
async def rewrite_chunk(chunk: str):
    examples = "\n\n".join(retrieve_human_examples())
    prompt = f"""
You are an expert human editor. Your task is to rewrite AI-generated text to make it indistinguishable from human writing while keeping the original meaning.

Rules:
- Use natural sentence variation, contractions, and smooth transitions
- Add minor rephrasing and synonyms to improve readability
- Rearrange sentences or combine short sentences naturally
- Keep the text suitable for students and academic essays
- Ensure proper grammar
- Length increase must not exceed 40%
- Mimic the style of the human examples below:

Human examples:
{examples}

AI Text:
{chunk}

Provide only the humanized text in your response.
"""
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.8,
                    "max_tokens": 2048
                },
            )
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print("LLM rewrite error:", e)
        return chunk

# ------------------ Post-Processing ------------------
def post_process(text: str):
    # Optional: minor synonym substitutions
    synonyms = {
        "important": "crucial", "significant": "notable",
        "show": "demonstrate", "powerful": "effective",
        "easy": "simple", "however": "though",
        "therefore": "so"
    }
    for k, v in synonyms.items():
        text = re.sub(rf"\b{k}\b", v, text, flags=re.IGNORECASE)

    # Contractions
    contractions = {
        "do not": "don't", "is not": "isn't", "can not": "can't",
        "will not": "won't", "does not": "doesn't", "should not": "shouldn't"
    }
    for k, v in contractions.items():
        text = re.sub(rf"\b{k}\b", v, text, flags=re.IGNORECASE)

    # Ensure length <=40% longer
    words = text.split()
    max_len = int(len(words) * 1.4)
    if len(words) > max_len:
        text = " ".join(words[:max_len])
    return text.strip()

async def process_chunk(chunk: str):
    rewritten = await rewrite_chunk(chunk)
    return post_process(rewritten)

# ------------------ API Endpoint ------------------
@app.post("/humanize")
async def humanize(req: HumanizeRequest):
    chunks = chunk_text(req.text)
    results = await asyncio.gather(*(process_chunk(c) for c in chunks))
    final_text = " ".join(results)
    orig_len = len(req.text.split())
    human_len = len(final_text.split())
    percent = round(((human_len - orig_len) / orig_len * 100), 1) if orig_len else 0
    return {
        "original_length": orig_len,
        "humanized_length": human_len,
        "percent_increase": percent,
        "output": final_text
    }

# ------------------ Serve Frontend ------------------
@app.get("/", response_class=FileResponse)
def serve_frontend():
    return FileResponse("index.html")
