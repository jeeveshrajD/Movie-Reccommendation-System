# Movie-Reccommendation-System
Movie Sync – Hybrid Movie Recommendation System

A Machine Learning project combining Collaborative Filtering (70%) and Content-Based Filtering (30%) for accurate and personalized movie recommendations.

Overview

Movie Sync is a hybrid movie recommendation engine designed to handle challenges like data sparsity, cold-start problems, and limited personalization commonly found in traditional recommender systems.

By combining the strengths of collaborative filtering and content-based filtering, this system delivers more accurate, diverse, and user-centric recommendations.
This project was developed as part of the Machine Learning Mini Project – VI Semester (ECE).

Features

🔢 Hybrid model

70% Collaborative Filtering

30% Content-Based Filtering

📊 Visualizations:

Data sparsity

Most-rated movies

User–item rating matrix

🧠 TF-IDF vectorization for metadata processing

🤝 KNN & Cosine Similarity for collaborative filtering

🎛 Gradio UI for interactive recommendations

🧹 Advanced preprocessing:

Metadata extraction

Tag generation

Cleaning & normalization

PROJECT STRUCTURE 
movie-recommendation-system/
│
├── src/
│   ├── hybrid_recommender.py          # Full hybrid model implementation
│
├── data/
│   ├── sample_movies.csv              # Small sample (full dataset not uploaded)
│   ├── sample_ratings.csv
│
├── README.md
├── requirements.txt
├── .gitignore
Dataset Information

This project uses publicly available datasets such as MovieLens and TMDB metadata, including:

movies.csv

ratings.csv

links.csv

tags.csv

keywords.csv

credits.csv

(Full dataset cannot be uploaded to GitHub due to file size limits.)

Download Full Dataset Here:
https://grouplens.org/datasets/movielens/

Methodology
1️⃣ Data Preprocessing

✔ Merging multiple CSV files
✔ Cleaning duplicates
✔ Handling missing values
✔ Extracting metadata:

cast

crew

genres

keywords
✔ Creating composite tags field (genres + keywords + descriptions)
✔ Text normalization
✔ Pivot table creation for ratings

2️⃣ Content-Based Filtering (30%)

Uses movie metadata to find similar films.

Steps:

Create tags column

Convert text into vectors using TF-IDF

Compute cosine similarity between movies

Recommend top similar movies

3️⃣ Collaborative Filtering (70%)

Uses user–item rating matrix to find patterns.

Steps:

Create pivot table: movies × users

Apply filtering to remove sparse movies

Use KNN (cosine distance)

Calculate nearest neighbors

Predict top rated items

Addresses:
✔ User preference learning
✔ Personalized recommendations

4️⃣ Hybrid Strategy (Our Final Model)

Uses weighted average:

final_score = 0.7 * collaborative_score + 0.3 * content_score


Benefits:

Solves cold-start problems

Reduces metadata reliance

Improves accuracy

Produces diverse suggestions

📈 Results & Discussion
✔ Collaborative Filtering Matrix

Revealed significant sparsity → filtering improved accuracy.
Dataset reduced from 6000+ movies to ~2100 after thresholding.

✔ Most Rated Movies

Popular movies (e.g., Star Wars, Braveheart) dominate user engagement.

✔ Hybrid Output

Hybrid model gives richer recommendations such as:

Content-based: Schindler’s List, Forrest Gump (emotional narrative)

Collaborative: Matrix, Terminator, Pulp Fiction

✔ Overall

Hybrid approach delivers highly relevant, diverse, and personalized suggestions.

🛠 Tech Stack

Python

pandas, numpy

scikit-learn

scipy

matplotlib

gradio

TF-IDF Vectorizer

KNN

🔮 Future Enhancements

✔ Adaptive (dynamic) weighting between CF & CBF
✔ Deep learning models (Autoencoders, Transformers)
✔ Better cold-start handling
✔ Real-time large-scale deployment
✔ User feedback loop for improving predictions


🏁 Conclusion

The hybrid Movie Sync system successfully merges collaborative and content-based approaches to deliver accurate, diverse, and user-centric movie recommendations. The combined strengths of both models overcome individual weaknesses and make this system a powerful foundation for future ML-based recommender systems.
