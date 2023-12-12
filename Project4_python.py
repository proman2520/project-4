import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# Load movies and ratings data
movies = pd.read_csv('Predictive model data/movies_title_reformatted.csv')
ratings = pd.read_csv('Predictive model data/ratings.csv')

# Pivot ratings data to create a movies-user matrix
moviesdb = ratings.pivot(index='movieId', columns='userId', values='rating')
moviesdb.fillna(0, inplace=True)

# Filter movies and users based on a threshold
nouser = ratings.groupby('movieId')['rating'].agg('count')
nomovies = ratings.groupby('userId')['rating'].agg('count')
moviesdb = moviesdb.loc[:, nomovies[nomovies > 50].index]
moviesdb = moviesdb.loc[nouser[nouser > 10].index]

# Convert the index to strings
moviesdb.index = moviesdb.index.astype(str)

# Create a sparse matrix
csr_data = csr_matrix(moviesdb.values)

# Reset the index
moviesdb.reset_index(inplace=True)

# Initialize KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)

# Separate the y variable, the labels
y = ratings['rating']

# Separate the X variable, the features
X = ratings.drop(columns=['rating'])

# Perform feature scaling (standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

# Initialize KNN regressor
knn_regressor = KNeighborsRegressor(n_neighbors=10)

# Fit the model to the training data
knn_regressor.fit(X_train, y_train)

# Make predictions on the test data
predictions = knn_regressor.predict(X_test)

# Calculate and print mean squared error
mse = mean_squared_error(predictions, y_test)
print(f"Mean Squared Error (Accuracy Score): {mse}")

def getrecs(movie):
#def getrecs(movieId):
    moviestorec = 10

    # Find the movieId of the given movie title
    mask = movies['title_reformatted'].str.upper() == movie.upper()
    movie_df = movies.loc[mask, 'title_reformatted']

    if movie_df.empty:
        print("Please check the spelling of the movie title or the movie may not be in our database :(")
        return

    movie_id = movies.loc[mask, 'movieId'].values
    movie_name = str(movie_df.values[0])  # Convert to string

    if len(movie_id) > 0:
        movie_id = movie_id[0]

        # Check if the movieId exists in the moviesdb DataFrame
        if str(movie_id) in moviesdb['movieId'].values:
            movieindex = moviesdb[moviesdb['movieId'] == str(movie_id)].index
            distances, indices = knn.kneighbors(csr_data[movieindex], n_neighbors=moviestorec + 1)
            recmovie = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
            recframe = []

            for val in recmovie:
                movieindex = moviesdb['movieId'].iloc[val[0]]
                released_year = int(movies[movies['movieId'] == int(movieindex)]['released_year'].values[0])
                recframe.append({
                    'Title': movies[movies['movieId'] == int(movieindex)]['title_reformatted'].values[0],
                    'Released Year': released_year,
                    'Distance': val[1]
                })
            rec_df = pd.DataFrame(recframe, index=range(1, moviestorec + 1))
            rec_df.sort_values(by='Distance', inplace=True)
            rec_df.reset_index(drop=True, inplace=True)
            print(f"If you enjoyed {movie_name} ({released_year}), here are the top 10 movies we think you'll also enjoy!")
            return rec_df
            #return rec_df[["Title", "Released Year"]]
        else:
            return print('There are not enough ratings for this movie.')
    else:
        return print('You get nothing you lose. Good day Sir!')
    
# Example usage of getrecs
getrecs('The other Guy')
getrecs('The Other Guys')
getrecs('IRON man')
getrecs('inCeption')
getrecs('Ted')
getrecs('Harry Potter and the Chamber of Secrets')
getrecs('Up')
getrecs('Avatar')
getrecs('Despicable Me')
# Remove duplicates in dataset so that it doesn't return an unexpected result? Getrecs('The Avengers') returns a movie from 1998.**
getrecs('the avengers')
getrecs('Django Unchained')
getrecs('Moana')
getrecs('Batman')

