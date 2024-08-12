# Movies-Rating-Predictions
This Repository is made for predicting user movie ratings using Gradient based Optimization 


---
# Movie Recommendation System Using Neural Collaborative Filtering

This project builds a movie recommendation system using Neural Collaborative Filtering (NCF) with TensorFlow and Keras. The model predicts user ratings for movies based on user and movie embeddings, trained on a subset of the MovieLens dataset.

## Project Overview

This project demonstrates the following steps:
1. **Data Loading**: Load and preprocess the MovieLens dataset.
2. **Data Preparation**: Encode the user and movie IDs and split the data into training, validation, and testing sets.
3. **Model Construction**: Create a neural network model to predict movie ratings based on user and movie embeddings.
4. **Model Training**: Train the model on the training data and validate it on the validation set.
5. **Model Evaluation**: Evaluate the model's performance on the test set.

## Directory Structure

```
.
├── Data/
│   ├── MovieLen/
│   │   ├── rating.csv
│   │   ├── movie.csv
│   └── ...
├── README.md
└── movie_recommendation_system.py
```

- `Data/`: Contains the MovieLens dataset files.
- `README.md`: This file, containing the project overview and instructions.
- `movie_recommendation_system.py`: The Python script implementing the recommendation system.

## Requirements

To run this project, you need the following packages:

- `numpy`
- `pandas`
- `matplotlib`
- `tensorflow`
- `keras`
- `sklearn`

You can install the required packages using `pip`:

```bash
pip install numpy pandas matplotlib tensorflow scikit-learn
```

## How to Run

1. **Data Preparation**: Ensure the `rating.csv` and `movie.csv` files are present in the `Data/MovieLen/` directory.
   
2. **Run the Script**:
   Execute the Python script to load data, prepare it, and train the model:

   ```bash
   python movie_recommendation_system.py
   ```

3. **Model Training and Evaluation**: The script will train two models with different embedding sizes (50 and 40) and evaluate their performance on the test set.

4. **Results**: The predictions from both models will be saved in a DataFrame and printed.

## Explanation of Key Components

### Data Loading

The script loads the movie ratings data (`rating.csv`) and movie metadata (`movie.csv`) from the MovieLens dataset.

### Data Preparation

- **Top-K Filtering**: Filters the top 15 users and movies based on the number of ratings.
- **Label Encoding**: Encodes `userId` and `movieId` to ensure they have continuous IDs.
- **Train-Test Split**: Splits the data into training, validation, and testing sets.

### Model Construction

Two neural network models are built with different embedding sizes:
- **User and Movie Embeddings**: These are dense representations learned for both users and movies.
- **Bias Terms**: User and movie biases are included to adjust the rating predictions.
- **Dot Product**: The dot product of user and movie embeddings is computed to capture the interaction between users and movies.

### Model Training

The models are trained using Mean Squared Error (MSE) as the loss function and Root Mean Squared Error (RMSE) as the evaluation metric.

### Model Evaluation

The predictions from the models are evaluated on the test set and compared with the actual ratings.

## Conclusion

This project provides a basic implementation of a neural collaborative filtering model for movie recommendations. You can extend it by experimenting with different network architectures, hyperparameters, and additional features.

---