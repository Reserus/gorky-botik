import pandas as pd
import pickle
import os
from sentence_transformers import SentenceTransformer

EMBEDDINGS_FILE = r"C:\Users\enoto\botik_serega\data\embeddings.pkl"
MODEL_NAME = 'paraphrase-MiniLM-L6-v2'

def load_data(path: str):
    df = pd.read_excel(path)
    df["longitude"] = df["coordinate"].apply(lambda s: float(s.split()[1][1:]))
    df["latitude"] = df["coordinate"].apply(lambda s: float(s.split()[2][:-1]))

    embeddings = None
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            embeddings = pickle.load(f)
        print("Loaded embeddings from cache.")
    else:
        print("Generating embeddings...")
        model = SentenceTransformer(MODEL_NAME)
        # Combine title and description for embedding
        texts_to_embed = (df['title'] + ". " + df['description'].fillna('')).tolist()
        embeddings = model.encode(texts_to_embed, show_progress_bar=True)
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(embeddings, f)
        print("Embeddings generated and saved to cache.")
    
    return df, embeddings
