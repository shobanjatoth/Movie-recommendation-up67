import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import faiss
import numpy as np
import pickle

# Load and vectorize data
data = pd.read_csv("main_data.csv", usecols=["movie_title", "comb"])
cv = CountVectorizer(max_features=5000)
count_matrix = cv.fit_transform(data["comb"]).toarray().astype("float32")

# Normalize for cosine-like similarity
count_matrix /= np.linalg.norm(count_matrix, axis=1, keepdims=True)

# Create FAISS index
index = faiss.IndexFlatIP(count_matrix.shape[1])  # Inner product = cosine similarity on normalized vectors
index.add(count_matrix)

# Save everything
faiss.write_index(index, "faiss_movie_index.index")
with open("movie_titles.pkl", "wb") as f:
    pickle.dump(list(data["movie_title"]), f)
