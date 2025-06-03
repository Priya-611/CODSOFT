#  Movie Rating Prediction using Random Forest in Python

This project predicts the **IMDb rating** of Indian movies using **Random Forest Regression**, based on features such as **genre, director, actors, year, duration, and votes**.
Project Goal: To analyze historical movie data and build a machine learning model that accurately predicts IMDb ratings, offering insights into which features (genre, director, cast, etc.) influence viewer ratings.


##  What is Random Forest?
- **Random**: Uses random subsets of data and features to build multiple decision trees.
- **Forest**: Combines the predictions of multiple trees to make more accurate and stable predictions.
- Ideal for **non-linear**, complex relationships where simple models might fail.



##  Dataset
[Kaggle](https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies)
The dataset used contains information about Indian movies such as:
- Name
- Year
- Duration
- Genre
- Director
- Actors
- IMDb Rating
- Votes



##  Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- scikit-learn (Random Forest, model evaluation metrics)
 

##  Data Preprocessing Steps

1. **Dropped irrelevant column**: `Name`
2. **Removed rows with missing Votes or Ratings**
3. **Handled missing numeric fields** (Duration → filled with mean)
4. **Converted string to numeric** (Votes, Year, Duration)
5. **Filled missing categorical fields** (`Genre`, `Director`, `Actors`) with `"Other"`
6. **One-hot encoded** categorical variables for modeling



##  Model Used

### Random Forest Regressor
- Captures complex non-linear patterns
- Ensemble method — combines many decision trees
- Automatically handles feature interactions



##  Evaluation Metrics

- **Mean Squared Error (MSE)**:  
  Measures average squared difference between actual and predicted ratings.  
  `MSE ≈ 1.18` (lower is better)

- **R² Score (R-squared)**:  
  Represents how much variance in ratings is explained by the model.  
  `R² ≈ 0.36` (higher is better; 1 = perfect)



## Sample Output

- **Mean_sqaures_error**: 1.1777727935606062
  **r2_Square**: 0.3608033059464323
