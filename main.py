from fastapi import FastAPI
from pydantic import BaseModel
import re
import random
from typing import List
from concurrent.futures import ThreadPoolExecutor
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Humanizer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class HumanizeRequest(BaseModel):
    text: str

def chunk_text(text: str, max_words: int = 200) -> List[str]:
    words = text.split()
    return [ " ".join(words[i:i+max_words]) for i in range(0, len(words), max_words) ]

# Expanded for student-like natural writing
STUDENT_SYNONYMS = {
    'important': ['key', 'crucial', 'pretty important'],
    'many': ['lots of', 'a bunch', 'several'],
    'help': ['help out', 'assist'],
    'use': ['use', 'try out'],
    'shows': ['shows', 'makes clear'],
    'however': ['but', 'though'],
    'furthermore': ['also', 'plus'],
}

HUMAN_FILLERS = ['you know', 'like', 'basically', 'I mean']
ROBOTIC_TO_CASUAL = {
    'In conclusion': 'So yeah',
    'Furthermore': 'Also',
    'In addition': 'Plus',
    'It is important to note': 'One thing is',
}

def process_chunk(chunk: str) -> str:
    # 1. Vary sentence length - split long ones
    sentences = re.split(r'(?<=[.!?]) +', chunk)
    varied = []
    for s in sentences:
        words = s.split()
        if len(words) > 20 and random.random() < 0.5:
            # Split long sentence
            mid = len(words) // 2 + random.randint(-2, 2)
            varied.append(' '.join(words[:mid]) + '.')
            varied.append(' '.join(words[mid:]))
        else:
            varied.append(s)
    
    text = ' '.join(varied)
    
    # 2. Synonym replacement & casual words
    words = text.split()
    for i, word in enumerate(words):
        clean_word = re.sub(r'[^\w]', '', word.lower())
        if clean_word in STUDENT_SYNONYMS and random.random() < 0.3:
            words[i] = random.choice(STUDENT_SYNONYMS[clean_word])
    text = ' '.join(words)
    
    # 3. Replace robotic transitions
    for robotic, casual in ROBOTIC_TO_CASUAL.items():
        if random.random() < 0.7:
            text = text.replace(robotic, casual)
    
    # 4. Contractions & imperfections
    contractions_map = {
        " do not ": " don't ",
        " does not ": " doesn't ",
        " is not ": " isn't ",
        " are not ": " aren't ",
        " it is ": " it's ",
    }
    for formal, casual in contractions_map.items():
        text = text.replace(formal, casual)
    
    # Occasional filler for human feel
    sentences = re.split(r'(?<=[.!?]) +', text)
    for i, s in enumerate(sentences):
        if random.random() < 0.2 and len(s.split()) > 8:
            filler = random.choice(HUMAN_FILLERS)
            s = s[:len(s)//2] + filler + s[len(s)//2:]
            sentences[i] = s
    text = ' '.join(sentences)
    
    # 5. Length control (max 30% increase)
    orig_len = len(chunk.split())
    words = text.split()
    if len(words) > orig_len * 1.3:
        text = ' '.join(words[:int(orig_len * 1.3)])
    
    return text

@app.post("/humanize")
def humanize(req: HumanizeRequest):
    if not req.text.strip():
        return {"original_length": 0, "humanized_length": 0, "output": ""}
    
    chunks = chunk_text(req.text)
    
    with ThreadPoolExecutor(max_workers=min(8, len(chunks))) as executor:
        humanized_chunks = list(executor.map(process_chunk, chunks))
    
    output = re.sub(r'\s+', ' ', ' '.join(humanized_chunks)).strip()
    
    orig_len = len(req.text.split())
    out_len = len(output.split())
    
    return {
        "original_length": orig_len,
        "humanized_length": out_len,
        "output": output
    }

@app.get("/")
def root():
    return {"status": "Humanizer API ready", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

