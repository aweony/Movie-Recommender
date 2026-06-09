import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "TMDB_movie_dataset_v11.csv")
TMDB_IMG_BASE = "https://image.tmdb.org/t/p/w500"

# Filter to released movies with enough votes and a non-empty overview
df = pd.read_csv(DATA)
df = df[df["status"] == "Released"]
df = df[df["vote_count"] >= 100]
df = df.dropna(subset=["overview"])
df = df.reset_index(drop=True)

for col in ["genres", "keywords", "tagline"]:
    df[col] = df[col].fillna("")

# Repeat genres/keywords so TF-IDF weights them more heavily than the overview
df["tags"] = (
    df["genres"] + " " + df["genres"] + " " +
    df["keywords"] + " " + df["keywords"] + " " +
    df["overview"] + " " +
    df["tagline"]
).str.lower()

# Bigrams capture phrases like "science fiction" and "based on comic"
tfidf = TfidfVectorizer(stop_words="english", max_features=10000, ngram_range=(1, 2))
tfidf_matrix = tfidf.fit_transform(df["tags"])

df["title_lower"] = df["title"].str.lower()


def _reason(query_genres: str, rec_genres: str, score: float) -> str:
    query_movies = {g.strip().lower() for g in query_genres.split(",")} - {""}
    rec_movies = {g.strip().lower() for g in rec_genres.split(",")} - {""}
    shared = query_movies & rec_movies
    if shared:
        label = ", ".join(sorted(shared)[:2]).title()
        plural = "s" if len(shared) > 1 else ""
        return f"Shares the {label} genre{plural} with your pick."
    if score > 0.4:
        return "Very similar themes and subject matter."
    return "Recommended based on overall content similarity."


def recommend(movie_title: str, n: int = 10) -> list | str:
    query = movie_title.strip().lower()

    # Exact title match, then fall back to substring match
    match = df[df["title_lower"] == query]
    if match.empty:
        match = df[df["title_lower"].str.contains(query, regex=False, na=False)]
    if match.empty:
        return f"No movie found matching '{movie_title}'."

    idx = match.index[0]
    query_row = df.iloc[idx]

    # Compute similarity for just this one movie — avoids a 1.3 GB precomputed matrix
    scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten() # use .flattern to turn result to a array of score
    scores[idx] = -1  # exclude the query movie itself

    top_indices = scores.argsort()[::-1][:n]

    results = []
    for i in top_indices:
        row = df.iloc[i]
        poster = row["poster_path"]
        results.append({
            "title": row["title"],
            "overview": row["overview"],
            "genres": row["genres"],
            "rating": round(float(row["vote_average"]), 1),
            "poster_url": f"{TMDB_IMG_BASE}{poster}" if pd.notna(poster) and poster else None,
            "release_year": str(row["release_date"])[:4] if pd.notna(row["release_date"]) else "N/A",
            "score": round(float(scores[i]), 3),
            "reason": _reason(query_row["genres"], row["genres"], float(scores[i])),
        })
    return results

