import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import requests
import re
from bs4 import BeautifulSoup
from tmdbv3api import TMDb, Movie
import faiss
import os
import time

# TMDb API setup
tmdb = TMDb()
tmdb.api_key = '72d19966f145a12485b7033ed0526058'  # Replace with your TMDb API key

# Load NLP model and vectorizer
clf = pickle.load(open('nlp_model1.pkl', 'rb'))
vectorizer = pickle.load(open('tranform1.pkl', 'rb'))

# Load FAISS index and movie titles
faiss_index = faiss.read_index("faiss_movie_index.index")
with open("movie_titles.pkl", "rb") as f:
    movie_titles = pickle.load(f)

# Helper to recommend movies
def rcmd(movie):
    movie = movie.lower()
    titles_lower = [title.lower() for title in movie_titles]
    if movie not in titles_lower:
        return 'Sorry! The movie you searched is not in our database. Please check the spelling or try with some other movies'

    idx = titles_lower.index(movie)

    # Vectorize and normalize the input
    vector = vectorizer.transform([movie_titles[idx]]).toarray().astype("float32")
    vector /= np.linalg.norm(vector, axis=1, keepdims=True)

    D, I = faiss_index.search(vector, 11)
    indices = I[0][1:]  # Exclude the movie itself

    return [movie_titles[i] for i in indices]

def ListOfGenres(genre_json):
    return ", ".join([g['name'] for g in genre_json]) if genre_json else "N/A"

def date_convert(s):
    MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    y = s[:4]
    m = int(s[5:7])
    d = s[8:10]
    return f"{MONTHS[m - 1]} {d}, {y}"

def MinsToHours(duration):
    return f"{duration // 60} hours {duration % 60} minutes" if duration else "N/A"

def get_suggestions():
    return list(map(str.capitalize, movie_titles))

def clean_review(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9.,!?'\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:300]

def get_trailer_url(movie_id):
    try:
        video_response = requests.get(
            f'https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={tmdb.api_key}'
        )
        videos = video_response.json().get('results', [])
        for video in videos:
            if video['type'] == 'Trailer' and video['site'] == 'YouTube':
                return f"https://www.youtube.com/embed/{video['key']}"
    except:
        return None
    return None

def get_poster_url(movie_id):
    response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={tmdb.api_key}')
    data_json = response.json()
    poster_path = data_json.get('poster_path', '')
    if poster_path:
        return f"https://image.tmdb.org/t/p/original{poster_path}"
    return "https://via.placeholder.com/300x450?text=No+Image"

app = Flask(__name__)

@app.route("/")
def home():
    suggestions = get_suggestions()
    return render_template('home.html', suggestions=suggestions)

@app.route("/recommend")
def recommend():
    movie = request.args.get('movie')
    if not movie:
        return "Movie parameter is missing!", 400

    r = rcmd(movie)
    movie = movie.upper()
    suggestions = get_suggestions()

    if isinstance(r, str):
        return render_template(
            'recommend.html',
            movie=movie,
            r=r,
            t='s',
            suggestions=suggestions,
            result=None,
            cards={},
            reviews={},
            img_path="",
            genres="N/A",
            vote_count="N/A",
            release_date="N/A",
            status="N/A",
            runtime="N/A",
            trailer_url=None
        )

    tmdb_movie = Movie()
    result = tmdb_movie.search(movie)
    if not result:
        return render_template(
            'recommend.html',
            movie=movie,
            r="Movie not found in TMDb!",
            t='s',
            suggestions=suggestions,
            result=None,
            cards={},
            reviews={},
            img_path="",
            genres="N/A",
            vote_count="N/A",
            release_date="N/A",
            status="N/A",
            runtime="N/A",
            trailer_url=None
        )

    movie_id = result[0].id
    movie_name = result[0].title

    try:
        response = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={tmdb.api_key}')
        data_json = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch main movie details: {e}")
        data_json = {}

    poster = data_json.get('poster_path', '')
    img_path = f'https://image.tmdb.org/t/p/original{poster}' if poster else ""
    genre = ListOfGenres(data_json.get('genres', []))
    trailer_url = get_trailer_url(movie_id)

    movie_reviews = {}
    try:
        review_response = requests.get(
            f'https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={tmdb.api_key}'
        )
        reviews_list = review_response.json().get('results', [])

        if reviews_list:
            for review in reviews_list:
                raw_content = review.get('content', '')
                author = review.get('author', 'Anonymous')
                created_at = review.get('created_at', '')[:10]
                rating = review.get('author_details', {}).get('rating', 'N/A')

                clean_text = clean_review(raw_content)
                if clean_text:
                    movie_vector = vectorizer.transform([clean_text])
                    pred = clf.predict(movie_vector)
                    sentiment = 'Good' if pred else 'Bad'
                    key = f'"{author}" on {created_at} (Rating: {rating})'
                    movie_reviews[key] = {'review': clean_text, 'sentiment': sentiment}
        else:
            movie_reviews = {"Notice": {"review": "No reviews available on TMDb.", "sentiment": "N/A"}}
    except:
        movie_reviews = {"Error": {"review": "Failed to fetch TMDb reviews.", "sentiment": "N/A"}}

    vote_count = "{:,}".format(result[0].vote_count)
    rd = date_convert(result[0].release_date)
    status = data_json.get('status', 'Unknown')
    runtime = MinsToHours(data_json.get('runtime', 0))

    movie_cards = {}
    for movie_title in r:
        try:
            list_result = tmdb_movie.search(movie_title)
            time.sleep(0.3)
            if not list_result:
                continue
            rec_id = list_result[0].id

            try:
                rec_response = requests.get(f'https://api.themoviedb.org/3/movie/{rec_id}?api_key={tmdb.api_key}')
                rec_data = rec_response.json()
            except requests.exceptions.RequestException as e:
                print(f"Failed to fetch recommended movie {movie_title}: {e}")
                continue

            poster_url = f"https://image.tmdb.org/t/p/original{rec_data.get('poster_path', '')}"
            rec_trailer = get_trailer_url(rec_id)

            movie_cards[movie_title] = {
                'poster': poster_url,
                'trailer': rec_trailer
            }

        except Exception as e:
            print(f"Search failed for recommended movie {movie_title}: {e}")
            continue

    return render_template('recommend.html',
                           movie=movie,
                           mtitle=r,
                           t='l',
                           cards=movie_cards,
                           result=result[0],
                           reviews=movie_reviews,
                           img_path=img_path,
                           genres=genre,
                           vote_count=vote_count,
                           release_date=rd,
                           status=status,
                           runtime=runtime,
                           trailer_url=trailer_url,
                           suggestions=suggestions)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)



