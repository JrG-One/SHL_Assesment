import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load data
df = pd.read_csv("shl_assessments_detailed.csv")
df['Description'] = df['Description'].astype(str)
df = df[df['Description'].str.lower().ne("nan") & df['Description'].str.strip().ne("")]
descriptions = df['Description'].tolist()

@st.cache_resource
def load_model_and_index():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(descriptions)
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))
    return model, index

model, index = load_model_and_index()

st.title("🔍 SHL Assessment Recommender")
st.write("Enter a job description or query to find matching assessments.")

query = st.text_area("Job Description or Query", height=200)
top_k = st.slider("Number of Recommendations", 5, 20, 10)

if st.button("Get Recommendations") and query.strip():
    with st.spinner("Thinking... 🤔"):
        query_embedding = model.encode([query])
        distances, indices = index.search(np.array(query_embedding), top_k)

        results = df.iloc[indices[0]].copy()
        results['Similarity Score'] = distances[0]
        results = results[['Assessment Name', 'Test Type', 'Duration', 'Remote Testing', 'Adaptive Support', 'Similarity Score', 'Link']]
        
        st.success("Here are your top matches:")
        st.dataframe(results.reset_index(drop=True))
        for _, row in results.iterrows():
            st.markdown(f"**{row['Assessment Name']}**  \n[{row['Link']}]({row['Link']})")
            st.markdown("---")
