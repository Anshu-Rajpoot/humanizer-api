from fastapi import FastAPI
from pydantic import BaseModel
import re
import random
from typing import List
from concurrent.futures import ThreadPoolExecutor
import nltk
from nltk.corpus import wordnet



# download once
nltk.download('wordnet')
nltk.download('omw-1.4')

app = FastAPI(title="AI Humanizer API")


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all (important for testing)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------- Request Schema --------
class HumanizeRequest(BaseModel):
    text: str


# -------- Core Functions --------

def chunk_text(text: str, max_words: int = 150) -> List[str]:
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]


def synonym_replace(word):
    synonyms = wordnet.synsets(word)
    if synonyms:
        lemmas = synonyms[0].lemma_names()
        if lemmas:
            return lemmas[0].replace("_", " ")
    return word


def random_synonym_injection(text: str) -> str:
    words = text.split()
    for i in range(len(words)):
        if random.random() < 0.25:
            words[i] = synonym_replace(words[i])
    return " ".join(words)


def structural_variation(text: str) -> str:
    sentences = re.split(r'(?<=[.!?]) +', text)

    # shuffle
    if len(sentences) > 3:
        random.shuffle(sentences)

    # break long sentences
    new_sentences = []
    for s in sentences:
        if len(s.split()) > 18 and random.random() > 0.5:
            parts = s.split(",")
            new_sentences.extend(parts)
        else:
            new_sentences.append(s)

    return " ".join(new_sentences)


def human_noise(text: str) -> str:
    fillers = ["I think", "honestly", "to be fair", "in a way", "you could say"]
    sentences = re.split(r'(?<=[.!?]) +', text)

    for i in range(len(sentences)):
        if random.random() < 0.3:
            sentences[i] = random.choice(fillers) + ", " + sentences[i]

    return " ".join(sentences)


def inject_variability(text: str) -> str:
    replacements = {
        "important": ["really important", "quite important"],
        "many": ["a lot of", "quite a few"],
        "shows": ["kind of shows", "basically shows"],
        "helps": ["can help", "tends to help"]
    }

    words = text.split()
    for i in range(len(words)):
        w = words[i].lower()
        if w in replacements and random.random() < 0.4:
            words[i] = random.choice(replacements[w])

    return " ".join(words)


def ai_score_heuristic(text: str) -> float:
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) < 2:
        return 0.5

    lengths = [len(s.split()) for s in sentences]
    avg = sum(lengths) / len(lengths)
    variance = sum((l - avg) ** 2 for l in lengths) / len(lengths)

    words = text.lower().split()
    unique_ratio = len(set(words)) / len(words)

    score = 0
    if variance < 20:
        score += 0.4
    if unique_ratio < 0.45:
        score += 0.3
    if avg > 18:
        score += 0.3

    return min(score, 1.0)


def process_chunk(chunk: str) -> str:
    best = chunk
    best_score = 1.0

    for _ in range(3):
        text = structural_variation(chunk)
        text = humanize_phrases(text)
        text = human_noise(text)
        text = inject_variability(text)

        score = ai_score_heuristic(text)

        if score < best_score:
            best_score = score
            best = text

        if score < 0.25:
            break

    return best


def humanize_phrases(text: str) -> str:
    phrase_map = {
        "In conclusion": ["To sum it up", "Overall", "At the end of the day"],
        "Furthermore": ["Also", "On top of that", "Besides"],
        "However": ["But", "That said", "Still"],
        "In addition": ["Plus", "Also", "Another thing is"],
        "It is important to note that": ["One thing to keep in mind is", "It's worth noting"],
        "This shows that": ["This basically shows", "This kind of means"],
        "There are many": ["There are quite a few", "You’ll find a lot of"],
        "In today's world": ["These days", "Right now"],
    }

    for phrase, variations in phrase_map.items():
        if phrase in text and random.random() < 0.6:
            text = text.replace(phrase, random.choice(variations))

    # contractions (VERY IMPORTANT for human feel)
    contractions = {
        "do not": "don't",
        "does not": "doesn't",
        "is not": "isn't",
        "are not": "aren't",
        "cannot": "can't",
        "it is": "it's",
        "that is": "that's"
    }

    for k, v in contractions.items():
        if k in text and random.random() < 0.5:
            text = text.replace(k, v)

    return text

# -------- API --------

@app.post("/humanize")
def humanize(req: HumanizeRequest):
    chunks = chunk_text(req.text)

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(process_chunk, chunks))

    final_text = " ".join(results)

    # length constraint
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