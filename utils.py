import spacy
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

nlp = spacy.load("en_core_web_sm")  # For chunking

def preprocess_and_chunk(text, chunk_size=200):
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
    # Split into paragraphs or fixed windows
    doc = nlp(text)
    chunks = []
    current_chunk = ""
    for sent in doc.sents:
        current_chunk += sent.text + " "
        if len(current_chunk.split()) > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = ""
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def compute_similarity(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

def generate_explanation(flag_type, chunk_idx, emotions_before, emotions_after, contradiction_details=None):
    if flag_type == "drift":
        dom_before = max(emotions_before, key=emotions_before.get)
        dom_after = max(emotions_after, key=emotions_after.get)
        return f"Tone shifts from {dom_before} to {dom_after} here, which may reduce audience trust."
    elif flag_type == "contradiction":
        return f"Contradiction detected: {contradiction_details}. This may confuse readers."
    elif flag_type == "confusion":
        return "High emotional variance in this section may lead to audience confusion."
    return "No issues detected."