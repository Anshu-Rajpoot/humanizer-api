# FastAPI Backend for AI Text Humanizer (FINAL - Assignment Ready)
# Requirements:
# pip install fastapi uvicorn transformers accelerate sentencepiece

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import re
import random
from concurrent.futures import ThreadPoolExecutor

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

app = FastAPI(title="AI Humanizer API")

# NOTE: For production, replace with llama.cpp or vLLM for speed
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype="auto"
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# -------- Request Schema --------
class HumanizeRequest(BaseModel):
    text: str

# -------- Utility Functions --------

def chunk_text(text: str, max_words: int = 150) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]


def structural_disruption(text: str) -> str:
    sentences = re.split(r'(?<=[.!?]) +', text)
    new_sentences = []

    for s in sentences:
        if len(s.split()) > 20 and random.random() > 0.5:
            parts = s.split(",")
            new_sentences.extend(parts)
        else:
            new_sentences.append(s)

    final = []
    i = 0
    while i < len(new_sentences):
        if i < len(new_sentences) - 1 and random.random() > 0.7:
            final.append(new_sentences[i] + " " + new_sentences[i + 1])
            i += 2
        else:
            final.append(new_sentences[i])
            i += 1

    return " ".join(final)


def shuffle_sentences(text: str) -> str:
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) > 3:
        random.shuffle(sentences)
    return " ".join(sentences)


def build_prompt(text: str) -> str:
    styles = [
        "a thoughtful student writing naturally",
        "a slightly informal but clear academic tone",
        "a human explaining ideas in a natural flow"
    ]
    style = random.choice(styles)

    return f"""
Rewrite the text below in a {style}.

IMPORTANT:
- Do NOT sound like AI
- Avoid predictable sentence patterns
- Mix short and long sentences randomly
- Occasionally break flow slightly like humans do
- Use natural phrasing, even slightly imperfect grammar
- Avoid repetitive structure
- Keep meaning intact
- Make it sound like genuine human writing

Text:
{text}

Rewritten:
"""


def human_noise(text: str) -> str:
    fillers = ["I think", "in a way", "you could say", "to be fair", "honestly"]
    sentences = re.split(r'(?<=[.!?]) +', text)

    for i in range(len(sentences)):
        if random.random() < 0.25:
            sentences[i] = random.choice(fillers) + ", " + sentences[i]

    return " ".join(sentences)


def inject_variability(text: str) -> str:
    replacements = {
        "important": ["really important", "quite important", "something important"],
        "shows": ["basically shows", "kind of shows", "clearly shows"],
        "helps": ["actually helps", "can help", "tends to help"],
        "many": ["a lot of", "quite a few", "many different"]
    }

    words = text.split()
    for i in range(len(words)):
        w = words[i].lower()
        if w in replacements and random.random() < 0.3:
            words[i] = random.choice(replacements[w])

    return " ".join(words)


def ai_score_heuristic(text: str) -> float:
    sentences = re.split(r'(?<=[.!?]) +', text)

    if len(sentences) < 2:
        return 0.5

    lengths = [len(s.split()) for s in sentences]
    avg_len = sum(lengths) / len(lengths)
    variance = sum((l - avg_len) ** 2 for l in lengths) / len(lengths)

    words = text.lower().split()
    unique_ratio = len(set(words)) / len(words)

    score = 0
    if variance < 20:
        score += 0.4
    if unique_ratio < 0.45:
        score += 0.3
    if avg_len > 18:
        score += 0.3

    return min(score, 1.0)


def aggressive_humanize(text: str) -> str:
    text = shuffle_sentences(text)
    text = structural_disruption(text)
    text = human_noise(text)
    text = inject_variability(text)
    return text


# -------- Core Pipeline --------

def process_chunk(chunk: str) -> str:
    chunk = structural_disruption(chunk)
    chunk = shuffle_sentences(chunk)

    best_output = ""
    best_score = 1.0

    for _ in range(3):
        prompt = build_prompt(chunk)

        output = generator(
            prompt,
            max_new_tokens=180,
            temperature=random.uniform(0.7, 1.1),
            top_p=random.uniform(0.85, 0.98),
            do_sample=True
        )[0]["generated_text"]

        rewritten = output.split("Rewritten:")[-1].strip()

        if "Text:" in rewritten:
            rewritten = rewritten.split("Text:")[-1].strip()

        rewritten = human_noise(rewritten)
        rewritten = inject_variability(rewritten)

        score = ai_score_heuristic(rewritten)

        if score < best_score:
            best_score = score
            best_output = rewritten

        if score < 0.35:
            break

    if best_score > 0.5:
        best_output = aggressive_humanize(best_output)

    return best_output


# -------- API Endpoint --------
@app.post("/humanize")
def humanize(req: HumanizeRequest):
    chunks = chunk_text(req.text)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_chunk, chunks))

    final_text = "\n".join(results)

    # Final humanization pass
    final_text = human_noise(final_text)
    final_text = inject_variability(final_text)

    # Enforce length constraint
    original_len = len(req.text.split())
    max_len = int(original_len * 1.4)

    words = final_text.split()
    if len(words) > max_len:
        final_text = " ".join(words[:max_len])

    return {
        "original_length": original_len,
        "humanized_length": len(final_text.split()),
        "output": final_text
    }


# -------- Health Check --------
@app.get("/")
def root():
    return {"status": "running"}


# -------- Run Locally --------
# uvicorn main:app --host 0.0.0.0 --port 8000
