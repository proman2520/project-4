# project-4 - Recommended for you and by you - Group 1 (Beatrice Apikos-Bennett, Andrew Prozorovsky, Shaji Qurashi, Patrick Moon)
For project 4, we decided to build a movie recommender. After entering a movie title, our predictive model uses item-based collaborative filtering to provide you a list of 10 movies we think you'll enjoy!

## What is item-based collaborative filtering? 

“Item based collaborative filtering was introduced 1998 by Amazon[6]. Unlike user based collaborative filtering, item based filtering looks at the similarity between different items, and does this by taking note of how many users that bought item X also bought item Y. If the correlation is high enough, a similarity can be presumed to exist between the two items, and they can be assumed to be similar to one another.”

## Steps to build our movie recommender

1. **Find data**: MovieLens is a wonderful and large dataset available for academic use. We are thankful they gave us permission to utilize their data!
2. **Clean data**:
  - CSVs fed into Postgres
  - Separated year from title and reformatted
  - Isolated genre instances 
  - Removed duplicates
  - Bulk formatted data in Postgres using DML
  - Queried data from Postgres to include necessary data and columns based on our needs
  - Assigned fake names to users (for fun) 
  - Etc…
3. **Build out**: Utilized k-nearest neighbors to find movies similar to the entry movie by rating across users and return 10 recommendations sorted by their similarity score.

## Technology used
- Postgres for storing the MovieLens data
- Google Cloud SQL for hosting Postgres instance
- Allowed collaboration and multiple users to access the same Postgres database
- SQLAlchemy to create a live connection to our cloud-hosted database and create charts
- Pandas and NumPy for model input data manipulation
- Scikit-Learn for building the predictive model
  - NearestNeighbors
  - KNeighborsRegressor
  - Mean Squared Error (MSE) for accuracy testing
- Tableau for visualizing metrics of our source data
- HTML and CSS for building a website

## Final note
Here is our repository for Project-4, in which we create an item-based collaborative filtering model to recommend 10 movies based on the movie of your choice from our dataset. The model is in working order, the plots render correctly, and our SQL database utilizes Google Cloud and the Python library SQLalchemy to share, manipulate, and utilize the data properly. The web UI properly explains our process in making the model and embeds the data's list of movies and our Tableau visualizations based on the dataset. The only non-functioning part of our project currently is the Flask app, which we would have used to spawn movie recommendations on the homepage of the web UI if we had more time. Fortunately, that is not a requirement of the project, but would have been a welcome addition with more time.

Thank you for grading and for a wonderful program which served to expand our portfolios, technical skills, and digital literacy. Cheers.

## Links
- Presentation: https://docs.google.com/presentation/d/1Lbg7oBn1TbQ5cX5So62mfE0Ae1lRmkOiqJ8aCpcrP2Q/edit#slide=id.gc6f889893_0_0
- Tableau story link: https://public.tableau.com/app/profile/shaji.qurashi/viz/Movies_17019978809680/Story1?publish=yes

## References
- MovieLens dataset: Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4: 19:1–19:19.
- https://www.geeksforgeeks.org/python-ways-to-sort-a-zipped-list-by-values/#
- https://www.geeksforgeeks.org/numpy-squeeze-in-python/
- https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/#:~:text=The%20K%2DNearest%20Neighbor%20 
- https://www.geeksforgeeks.org/user-based-collaborative-filtering/
- https://medium.com/grabngoinfo/recommendation-system-user-based-collaborative-filtering-a2e76e3e15c4
- https://realpython.com/build-recommendation-engine-collaborative-filtering/
- https://www.diva-portal.org/smash/get/diva2:1111865/FULLTEXT01.pdf
- https://movielens.org/
- https://doi.org/10.1145/2827872
- https://www.statology.org/valueerror-unknown-label-type-continuous/
- https://datascience.stackexchange.com/questions/20199/train-test-split-error-found-input-variables-with-inconsistent-numbers-of-sam
- https://www.ssa.gov/oact/babynames/
- ChatGPT for debugging predictive model
