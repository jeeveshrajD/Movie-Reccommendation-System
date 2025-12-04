import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
!pip install gradio
import gradio as gr

# Load datasets
movies = pd.read_csv("/content/drive/MyDrive/movies.csv")
ratings = pd.read_csv("/content/drive/MyDrive/ratings.csv")

print("\nLoaded Data:")
print(f"Movies Shape: {movies.shape}, Ratings Shape: {ratings.shape}")

# Data Sparsity Visualization Function
def plot_sparsity(dataset, title):
    plt.figure(figsize=(10, 6))
    plt.spy(dataset, markersize=1)
    plt.title(title)
    plt.xlabel("Users")
    plt.ylabel("Movies")
    plt.show(block=True)

# Collaborative Filtering Preprocessing
final_dataset = ratings.pivot(index="movieId", columns="userId", values="rating").fillna(0)

print("\nCreated Pivot Table for Collaborative Filtering:")
print(final_dataset.head())

# Plot BEFORE Filtering
plot_sparsity(final_dataset, "Before Filtering")

# Filter movies & users with minimal interactions
no_user_voted = ratings.groupby("movieId")['rating'].count()
no_movies_voted = ratings.groupby("userId")['rating'].count()
print(f"\nMovies before filtering: {len(final_dataset)}")
final_dataset = final_dataset.loc[no_user_voted[no_user_voted > 10].index, :]
final_dataset = final_dataset.loc[:, no_movies_voted[no_movies_voted > 50].index]
print(f"Movies after filtering: {len(final_dataset)}")

# Plot AFTER Filtering
plot_sparsity(final_dataset, "After Filtering")

# Train KNN Model for Collaborative Filtering
csr_data = csr_matrix(final_dataset.values)
final_dataset.reset_index(inplace=True)
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

print("\nKNN Model Trained for Collaborative Filtering")

# Content-Based Filtering Preprocessing
tfidf = TfidfVectorizer(stop_words='english')
movies['title'] = movies['title'].fillna('')
tfidf_matrix = tfidf.fit_transform(movies['title'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print("\nTF-IDF Matrix Created for Content-Based Filtering:")
print(f"Shape of TF-IDF Matrix: {tfidf_matrix.shape}")

# Hybrid Recommendation Function
def get_hybrid_recommendation(movie_name):
    movie_list = movies[movies['title'].str.contains(movie_name, case=False, na=False)]

    if movie_list.empty:
        return pd.DataFrame([["Movie not found", "N/A"]], columns=["Title", "Score"])

    movie_idx = movie_list.iloc[0]['movieId']
    print(f"\nSearching for Movie: {movie_name}, Movie ID: {movie_idx}")

    # Collaborative Filtering Recommendations
    matching_movies = final_dataset[final_dataset['movieId'] == movie_idx].index
    collaborative_recommendations = {}

    if len(matching_movies) > 0:
        movie_index = matching_movies[0]
        distances, indices = knn.kneighbors(csr_data[movie_index], n_neighbors=11)

        print("\nCollaborative Filtering Recommendations:")
        for val in zip(indices.squeeze().tolist(), distances.squeeze().tolist()):
            collab_movie_id = final_dataset.iloc[val[0]]['movieId']
            title = movies[movies['movieId'] == collab_movie_id]['title'].values[0]
            similarity_score = round(1 - val[1], 4)
            collaborative_recommendations[title] = similarity_score
            print(f" - {title} (Score: {similarity_score})")

    # Content-Based Filtering Recommendations
    content_index = movies[movies['movieId'] == movie_idx].index[0]
    content_scores = list(enumerate(cosine_sim[content_index]))
    content_scores = sorted(content_scores, key=lambda x: x[1], reverse=True)[1:11]

    content_recommendations = {}
    print("\nContent-Based Filtering Recommendations:")
    for i in content_scores:
        title = movies.iloc[i[0]]['title']
        content_recommendations[title] = round(i[1], 4)
        print(f" - {title} (Score: {round(i[1], 4)})")

    # Combining Collaborative & Content-Based Scores
    combined_recommendations = {}
    for movie in set(collaborative_recommendations.keys()).union(set(content_recommendations.keys())):
        collab_score = collaborative_recommendations.get(movie, 0) * 0.7
        content_score = content_recommendations.get(movie, 0) * 0.3
        combined_recommendations[movie] = round(collab_score + content_score, 4)

    # Filter out input movie
    sorted_recommendations = sorted(combined_recommendations.items(), key=lambda x: x[1], reverse=True)
    filtered_recommendations = [rec for rec in sorted_recommendations if rec[0].lower() != movie_list.iloc[0]['title'].lower()]

    # Convert to DataFrame and return top 10 recommendations
    df = pd.DataFrame(filtered_recommendations, columns=['Title', 'Score'])
    return df.head(10)

# Most Rated Movies Visualization
def plot_most_rated_movies():
    most_rated = ratings.groupby("movieId")['rating'].count().sort_values(ascending=False).head(10)
    most_rated_movies = movies[movies['movieId'].isin(most_rated.index)]

    plt.figure(figsize=(10, 6))
    plt.barh(most_rated_movies['title'], most_rated.values, color="purple")
    plt.xlabel("Number of Ratings")
    plt.ylabel("Movies")
    plt.title("Top 10 Most Rated Movies")
    plt.gca().invert_yaxis()
    plt.show(block=True)

# Show Most Rated Movies Before Running the Gradio App
plot_most_rated_movies()

# Gradio Interface for Recommendations
def recommend_movies(movie_name):
    df = get_hybrid_recommendation(movie_name)
    return df

app = gr.Interface(
    fn=recommend_movies,
    inputs="text",
    outputs=gr.Dataframe(headers=["Title", "Score"]),
    title="Hybrid Movie Recommendation System",
    description="Enter a movie name to get recommendations (70% Collaborative, 30% Content-Based)"
)

app.launch()
