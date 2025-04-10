# 🧠 SHL Assessment Recommender

A semantic search-based tool that recommends the most relevant SHL assessments based on a job description or query.

## 🔗 Live Demo
- Streamlit UI: [https://your-app.streamlit.app](#)
- FastAPI API: [https://shl-api.onrender.com/recommend](#)

## 💡 How It Works
- Scrapes SHL's official product catalog
- Cleans and enriches assessment data
- Uses `sentence-transformers` to create text embeddings
- Performs similarity search using `FAISS`
- Frontend: Streamlit | Backend API: FastAPI

## 🧪 Evaluation
Supports evaluation metrics like:
- `Recall@k`
- `MAP@k`

## 🚀 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py  # For UI
uvicorn api:app --reload  # For API
