from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import os

model = SentenceTransformer('distilbert-base-nli-mean-tokens')

def load_act_data(act_name):
    act_files = {
        "ROB": "data/ACT_ROB_1956.csv",
        "ROC": "data/ACT_ROC_2016.csv",
        "LLP": "data/ACT_LLP_2024.csv"
    }
    file_path = act_files.get(act_name.upper())
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No CSV found for {act_name}")
    return pd.read_csv(file_path)

def retrieve_context(query, act_name):
    index_path = f"data/embeddings/{act_name.lower()}_embeddings.faiss"
    df = load_act_data(act_name)
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"No FAISS index found for {act_name}")

    index = faiss.read_index(index_path)
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec), k=3)
    context = " ".join(df.iloc[i]["Content"] for i in I[0])
    return context
