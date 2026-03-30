from fastapi import FastAPI
from pydantic import BaseModel
import re
import random
from typing import List
from concurrent.futures import ThreadPoolExecutor
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Humanizer API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HumanizeRequest(BaseModel):
    text: str


# -------- TEXT PROCESSING --------

def chunk_text(text: str, max_words: int = 150) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]


# ✅ Lightweight synonym map (FAST)
SYN_MAP = {
    "important": ["key", "crucial"],
    "many": ["a lot of", "several"],
    "help": ["assist", "support"],
    "use": ["utilize", "apply"],
    "shows": ["demonstrates", "reveals"],
    "big": ["large", "major"]
}

def replace_words(text: str):
    words = text.split()
    for i in range(len(words)):
        w = words[i].lower()
        if w in SYN_MAP and random.random() < 0.3:
            words[i] = random.choice(SYN_MAP[w])
    return " ".join(words)


def structural_variation(text: str):
    sentences = re.split(r'(?<=[.!?]) +', text)

    # slight shuffle (not aggressive)
    if len(sentences) > 4 and random.random() > 0.5:
        random.shuffle(sentences)

    new = []
    for s in sentences:
        if len(s.split()) > 20 and random.random() > 0.6:
            parts = s.split(",")
            new.extend(parts)
        else:
            new.append(s)

    return " ".join(new)


def human_tone(text: str):
    replacements = {
        "In conclusion": "Overall",
        "Furthermore": "Also",
        "However": "But",
        "In addition": "Also",
        "It is important to note that": "One thing to keep in mind is"
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text


def contractions(text: str):
    pairs = {
        "do not": "don't",
        "does not": "doesn't",
        "is not": "isn't",
        "are not": "aren't",
        "cannot": "can't",
        "it is": "it's"
    }

    for k, v in pairs.items():
        if random.random() < 0.5:
            text = text.replace(k, v)

    return text


def ai_score_heuristic(text: str):
    sentences = re.split(r'(?<=[.!?]) +', text)
    lengths = [len(s.split()) for s in sentences if s]

    if not lengths:
        return 1.0

    avg = sum(lengths) / len(lengths)
    variance = sum((l - avg) ** 2 for l in lengths) / len(lengths)

    score = 0
    if variance < 15:
        score += 0.4
    if avg > 18:
        score += 0.3
    if len(set(text.split())) / len(text.split()) < 0.5:
        score += 0.3

    return min(score, 1.0)


def process_chunk(chunk: str):
    best = chunk
    best_score = 1.0

    for _ in range(3):
        text = structural_variation(chunk)
        text = replace_words(text)
        text = human_tone(text)
        text = contractions(text)

        score = ai_score_heuristic(text)

        if score < best_score:
            best_score = score
            best = text

        if score < 0.3:
            break

    return best


# -------- API --------

@app.post("/humanize")
def humanize(req: HumanizeRequest):
    chunks = chunk_text(req.text)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_chunk, chunks))

    final_text = " ".join(results)

    # length control (STRICT)
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


@app.get("/")
def root():
    return {"status": "running"}