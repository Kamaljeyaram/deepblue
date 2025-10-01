# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import os
import json
import requests
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# -------------------------------
# Config
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
SIMILARITY_THRESHOLD = 0.6

# -------------------------------
# Load categories JSON
with open("categories.json", "r") as f:
    categories = json.load(f)

# Initialize embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Build FAISS index
data = []
labels = []
for cat, examples in categories.items():
    for ex in examples:
        vec = embedder.encode(ex)
        data.append(vec / np.linalg.norm(vec))  # normalize
        labels.append(cat)

d = len(data[0])
index = faiss.IndexFlatIP(d)
index.add(np.array(data))

# -------------------------------
# Web search function
def web_search(query: str) -> str:
    url = "https://api.tavily.com/search"
    headers = {"Content-Type": "application/json"}
    payload = {"api_key": TAVILY_API_KEY, "query": query, "num_results": 3}
    resp = requests.post(url, headers=headers, data=json.dumps(payload))
    results = resp.json()
    contexts = [r["content"] for r in results.get("results", [])]
    return " ".join(contexts) if contexts else ""

# -------------------------------
# Gemini classification
def generate_category(transaction: str, context: str) -> str:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-Goog-Api-Key": GEMINI_API_KEY
    }

    prompt = f"""You are a financial transaction categorization assistant.
Categories: Food, Travel, Bills, Shopping, Entertainment, Health, Education
Transaction: "{transaction}"
Retrieved context:
{context}
Classify the transaction into one of the categories. Answer only with the category name.
"""

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    result = response.json()

    try:
        return result["candidates"][0]["content"]["parts"][0]["text"]
    except:
        return "Error in response"

# -------------------------------
# RAG categorization
def rag_categorize(transaction: str) -> str:
    q_vec = embedder.encode(transaction)
    q_vec = q_vec / np.linalg.norm(q_vec)
    D, I = index.search(np.array([q_vec]), 3)
    top_labels = [labels[i] for i in I[0]]
    top_scores = D[0]

    if max(top_scores) >= SIMILARITY_THRESHOLD:
        context = ", ".join(top_labels)
    else:
        context = web_search(transaction)

    category = generate_category(transaction, context)
    return category

# -------------------------------
# FastAPI setup
app = FastAPI(title="RAG Transaction Categorizer")

class Transaction(BaseModel):
    transaction: str

@app.post("/categorize")
def categorize(payload: Transaction):
    category = rag_categorize(payload.transaction)
    return {"category": category}
