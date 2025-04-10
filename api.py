from fastapi import FastAPI, Request
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np

# Load dataset
df = pd.read_csv("shl_assessments_detailed.csv")
df['Description'] = df['Description'].astype(str)
df = df[df['Description'].str.lower().ne("nan") & df['Description'].str.strip().ne("")]
descriptions = df['Description'].tolist()

# Load model + build FAISS index
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(descriptions)
index = faiss.IndexFlatL2(embeddings[0].shape[0])
index.add(np.array(embeddings))

# FastAPI setup
app = FastAPI()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

@app.post("/recommend")
def recommend_assessments(request: QueryRequest):
    query_embedding = model.encode([request.query])
    distances, indices = index.search(np.array(query_embedding), request.top_k)

    results = df.iloc[indices[0]].copy()
    results['score'] = distances[0]

    response = results[[
        "Assessment Name", "Test Type", "Duration", "Remote Testing",
        "Adaptive Support", "Link", "score"
    ]].reset_index(drop=True).to_dict(orient="records")
    
    return {"results": response}
