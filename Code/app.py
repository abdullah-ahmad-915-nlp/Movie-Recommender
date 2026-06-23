import os

if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
    del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

os.environ["GOOGLE_API_USE_CLIENT_CERTIFICATE"] = "false"


import streamlit as st
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


movies_df = pd.read_parquet("train_movies_df.parquet")
val_df     = pd.read_parquet("val_movies_df.parquet")
test_df    = pd.read_parquet("test_movies_df.parquet")

# FULL dataset (train+val+test)
movies_full = pd.concat([movies_df, val_df, test_df]).reset_index(drop=True)

# BUILD MOVIE CATALOG FOR RETRIEVAL
movies_full["combined_text"] = (
    movies_full["Title"].astype(str) + " " +
    movies_full["Overview"].astype(str) + " " +
    movies_full["reviewText"].astype(str)
)

movie_catalog = (
    movies_full.groupby("id", as_index=False)
    .agg(
        Title=("Title", lambda x: x.iloc[0]),
        Genre=("Genre", lambda x: " ".join(sorted(set(" ".join(x).split())))),
        combined_text=("combined_text", lambda x: " ".join(x)),
        avg_score=("numeric_score_out_of_10", "mean"),
        n_reviews=("numeric_score_out_of_10", "size")
    )
).reset_index(drop=True)


# LOAD EMBEDDING + CROSS ENCODER MODELS
st.write("Loading embedding & cross-encoder models...")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

movie_embeddings = model.encode(
    movie_catalog["combined_text"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True,
    batch_size=64,
    normalize_embeddings=True
)

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# RECOMMENDER FUNCTIONS
def apply_filters(df, user_query):
    q = user_query.lower()

    if "bad" in q or "worst" in q:
        df = df[df["avg_score"] < 4.5]
    if "good" in q or "best" in q or "highly rated" in q:
        df = df[df["avg_score"] >= 7.0]

    for g in ["horror", "comedy", "romance", "thriller", "action", "drama"]:
        if g in q:
            df = df[df["Genre"].str.contains(g, na=False)]
    return df


def smart_movie_recommender(user_query, top_k=10, candidate_pool=300):
    query_emb = model.encode([user_query], normalize_embeddings=True)[0]
    sims = movie_embeddings @ query_emb

    top_idx = np.argsort(sims)[::-1][:candidate_pool]
    candidates = movie_catalog.iloc[top_idx].copy()
    candidates["similarity"] = sims[top_idx]

    candidates = apply_filters(candidates, user_query)
    if len(candidates) == 0:
        candidates = movie_catalog.iloc[top_idx].copy()

    text_pairs = [(user_query, t) for t in candidates["combined_text"]]
    scores = cross_encoder.predict(text_pairs)
    candidates["cross_score"] = scores

    return candidates.sort_values("cross_score", ascending=False).head(top_k)


def summarize_reviews(text, max_len=2):
    sentences = [s.strip() for s in text.split(".")]
    long_sentences = []
    for s in sentences:
        if len(s) > 40:
            long_sentences.append(s)

    if len(long_sentences) == 0:
        return "Reviewers did not provide substantial feedback."

    return ". ".join(long_sentences[:max_len]) + "."


# LLM SETUP (LANGCHAIN + GEMINI)
if "GOOGLE_API_KEY" not in os.environ:
    st.warning("Please set your Google API key in the sidebar.")

# Do it this way to avoid a streamlit error for key initialization
def get_llm():
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None

    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        api_key=api_key,
        temperature=0.2,
    )

recommendation_prompt = PromptTemplate(
    template="""
You are a movie recommendation assistant.

The user asked: "{user_query}"

Below is structured information about the recommended movies:
{movie_data}

Write a detailed, conversational, human-like recommendation paragraph.
Requirements:
- Mention each movie by name.
- Explain why it fits the user's query.
- Include what reviewers generally said.
- Write ONLY in plain text.
""",
    input_variables=["user_query", "movie_data"]
)

parser = StrOutputParser()

# Also for streamlit - lazy setup essentially
def run_llm(user_query, movie_data):
    llm = get_llm()
    if llm is None:
        return " Please enter your Google API key in the sidebar."

    chain = recommendation_prompt | llm | parser
    return chain.invoke({"user_query": user_query, "movie_data": movie_data})


def build_llm_context(results):
    entries = []
    for _, row in results.iterrows():
        reviews = movies_full[movies_full["Title"] == row["Title"]]["reviewText"]
        summary = summarize_reviews(" ".join(reviews.tolist()))

        entries.append({
            "title": row["Title"].title(),
            "genre": row["Genre"],
            "avg_score": float(row["avg_score"]),
            "review_summary": summary,
        })
    return entries


def generate_llm_paragraph(user_query, results):
    movie_data = build_llm_context(results)

    block = ""
    for m in movie_data:
        block += (
            f"- Title: {m['title']}\n"
            f"  Genre: {m['genre']}\n"
            f"  Avg Score: {m['avg_score']}\n"
            f"  Review Summary: {m['review_summary']}\n\n"
        )

    return run_llm(user_query, block)


# STREAMLIT UI
st.title("Intelligent Movie Recommendation System")
st.write("Enter a natural-language description, genre, or preference.")

with st.sidebar:
    st.header("🔑 API Key")
    api_key = st.text_input("Google API Key", type="password")

    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("API Key saved! You may now run queries.")


# USER INPUT
user_query = st.text_area("Describe what kind of movie you're looking for:", height=100)

top_n = st.slider("Number of recommendations:", 1, 10, 5)

if st.button("Recommend Movies"):
    if not user_query.strip():
        st.error("Please enter a query.")
    else:
        with st.spinner("Finding the best movies for you..."):
            results = smart_movie_recommender(user_query, top_k=top_n)
            explanation = generate_llm_paragraph(user_query, results)

        st.subheader("Recommendation Summary")
        st.write(explanation)

        st.subheader("Top Movies")
        st.dataframe(results)
