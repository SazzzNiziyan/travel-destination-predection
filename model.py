# model.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load data
ratings = pd.read_csv("ratings.csv")

# Create pivot table
pivot_table = ratings.pivot_table(index='userId', columns='destination', values='rating').fillna(0)

# Compute cosine similarity
similarity = cosine_similarity(pivot_table.T)
similarity_df = pd.DataFrame(similarity, index=pivot_table.columns, columns=pivot_table.columns)

def recommend(destination, top_n=1):
    if destination not in similarity_df.columns:
        return "Destination not found"

    # Get similarity scores
    scores = similarity_df[destination].sort_values(ascending=False)
    scores = scores.drop(destination)  # Remove self
    return scores.index[:top_n].tolist()[0]  # Return top recommendation
