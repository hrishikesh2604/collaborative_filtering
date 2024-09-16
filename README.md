# Collaborative Filtering on MovieLens 20M Dataset

## Project Overview

This project demonstrates the application of **Collaborative Filtering** for movie recommendation using the **MovieLens 20-million dataset**. Collaborative filtering is a powerful technique widely used for recommendation systems, particularly in predicting user preferences for items (movies, products, etc.) based on user-item interaction history.

The dataset used in this project contains **20 million ratings** and **tag applications** applied to **27,000 movies** by **138,000 users**. This large-scale dataset allows for building robust recommendation systems using matrix factorization techniques and other collaborative filtering algorithms.

### Link to the Project:
[Collaborative Filtering on MovieLens 20M Dataset](https://www.kaggle.com/code/hrishikeshdongre2604/collaborative-filtering)

---

## Dataset

The **MovieLens 20M dataset** consists of the following components:

- **Ratings**: The core data consisting of users rating movies on a scale from 1 to 5.
- **Movies**: Information about movies including titles and genres.
- **Tags**: Tags that users have applied to movies, providing additional information about the content.
- **Links**: Links to external websites such as IMDb for additional metadata.

The dataset is ideal for developing recommendation systems due to its size and detailed user-item interaction data.

---

## Technologies Used

- **Python**: The primary programming language used for data preprocessing and model development.
- **Pandas**: For data manipulation, cleaning, and transformation.
- **NumPy**: Used for numerical operations, especially for matrix manipulations.
- **Surprise**: A Python library specifically designed for building and analyzing recommendation systems using collaborative filtering algorithms.
- **Matplotlib / Seaborn**: For visualizing trends and results.
- **Scikit-learn**: Used for evaluation metrics and splitting data into training and test sets.

---

## Collaborative Filtering Techniques Used

The project explores two major approaches within collaborative filtering:

### 1. **User-Based Collaborative Filtering**
   - This method recommends items based on the similarity between users. Users with similar taste (based on the items they have rated) are clustered, and their preferences are used to make predictions for unseen items.
   - **Key Idea**: If user A and user B have similar ratings for a set of movies, we can use user A’s ratings to predict user B’s ratings for unseen movies.

### 2. **Item-Based Collaborative Filtering**
   - This method recommends items based on the similarity between items. Movies that have been rated similarly by multiple users are considered similar, and those similarities are used to make predictions.
   - **Key Idea**: If a user likes movie A, and movie B is similar (based on user ratings), the system recommends movie B.

### 3. **Matrix Factorization (SVD - Singular Value Decomposition)**
   - Matrix factorization is a more advanced technique that transforms both users and items into a latent feature space. By breaking down the user-item matrix into lower-dimensional matrices, it captures underlying patterns in the data.
   - **SVD** (Singular Value Decomposition) is one of the most popular matrix factorization techniques used for recommendation. It identifies the latent features that explain user preferences and item characteristics, leading to accurate predictions.

---

## Key Steps in the Project

1. **Loading and Preprocessing the Data**
   - The dataset is loaded, and necessary preprocessing steps are applied, such as handling missing values, filtering out infrequent users/items, and creating train-test splits.

2. **Exploratory Data Analysis (EDA)**
   - Visualizations and analyses to understand the distribution of ratings, popular movies, active users, and the sparsity of the dataset.

3. **Building the Collaborative Filtering Model**
   - **User-based and item-based collaborative filtering models** are implemented using similarity measures such as cosine similarity or Pearson correlation.
   - **Matrix factorization with SVD** is employed to reduce the dimensionality of the user-item matrix and predict missing ratings.

4. **Model Training and Evaluation**
   - The models are trained on a subset of the data, and evaluation is done using metrics such as **Root Mean Squared Error (RMSE)**, **Mean Absolute Error (MAE)**, and **Precision@K/Recall@K**.
   - Hyperparameter tuning is done to improve the performance of the models.

5. **Making Predictions and Recommendations**
   - The trained models are used to predict ratings for movies that a user hasn’t seen. Based on these predicted ratings, personalized movie recommendations are generated for each user.

---

## How to Run the Project

1. **Clone the repository or download the notebook** from [Kaggle](https://www.kaggle.com/code/hrishikeshdongre2604/collaborative-filtering).
2. **Install required libraries**:
   ```bash
   pip install numpy pandas scikit-learn surprise matplotlib seaborn
