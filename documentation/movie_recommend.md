# Movie Recommendation System

## Objective
The goal of this project is to build a simple movie recommendation system based on user ratings. By analyzing the ratings and similarities between movies, the system can suggest movies that are similar to a given movie. The recommendations are based on the correlation between user ratings for different movies.

## Dataset
The dataset used consists of:
1. **u.data.txt**: Contains user ratings for different movies. The columns are:
   - **user_id**: Unique identifier for the user.
   - **item_id**: Unique identifier for the movie.
   - **rating**: The rating given by the user (1-5 scale).
   - **timestamp**: Timestamp of the rating.
   
2. **Movie_Id_Titles.csv**: Contains movie titles corresponding to the `item_id`.

### Data Preprocessing:
- The `u.data.txt` file is loaded and merged with the movie titles from the `Movie_Id_Titles.csv` file using the `item_id`.
- The dataset is cleaned and prepared for further analysis.

## Exploratory Data Analysis (EDA)

### Rating Distribution:
- **Average Rating per Movie**: Movies are grouped by title and their average rating is calculated.
- **Number of Ratings per Movie**: The number of ratings for each movie is computed.
- Visualized the distribution of:
  - Number of ratings per movie.
  - Average ratings per movie.

### Relationships:
- **Joint Plot**: Explored the relationship between the number of ratings and average rating.
  
## Movie Recommendation

### Creating the Movie Matrix:
- A pivot table is created with **user_id** as rows, **movie title** as columns, and **rating** as values.
  
### Movie Similarity:
- For each movie, the system computes the **correlation** between its user ratings and those of other movies. This measures the similarity between movies based on user preferences.
- **Example movies** used for correlation:
  - **Star Wars (1977)**
  - **Liar Liar (1997)**
  
### Results:
- The system identifies movies that are similar to **Star Wars (1977)** and **Liar Liar (1997)** based on user ratings.
- The top correlated movies for each of these two movies were identified, filtered by a minimum of 100 ratings.

## Conclusion
This project demonstrates the basic idea behind a movie recommendation system using collaborative filtering. By analyzing the user ratings and calculating the correlation between movies, it can suggest similar movies to a user. Future improvements can include using more advanced algorithms, such as matrix factorization or deep learning models, to provide more accurate recommendations.
