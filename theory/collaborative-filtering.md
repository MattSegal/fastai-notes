### Collaborative Filtering

A good canonical dataset for collborative filtering is the movielens dataset, in each row of the dataset, user X watched movie Y and gave it some rating out of 5. There is a many-to-many relationship between movies and users. We can also see the title and genres of the movies.

For some user / movie combo that we haven't seen before, we need to predict how much the user will like the movie.

#### Linear Collaborative Filtering

We can take a simple, linear approach to this problem (_probabalistic matrix factorization_).

We have `n` users, `m` movies, and `n x m` possible ratings. We can construct a `n x m` matrix of ratings `R`. Some of the values of this matrix will be blank, since we have not receive ratings for some (user, movie) pairs.

We can select an arbitrary embedding size, say `e = 5`, which we can use to create a `n x e` embedding matrix for users `U` and a `e x m` embedding matrix `M` for movies. We can calculate our predicted ratings matrix `R_hat`, where `M x U = R_hat`. We can use a root-mean-squared-error loss function for the difference between `R` and `R_hat`, then use gradient descent to solve for the values of `M` and `U`. Importantly, we shouldn't try to learn from the blank values - I'm not sure exactly how to implement that. Once we're done fitting our model, we should be able to predict the rating of user `i` on movie `j` by calculating `R_hat[i][j]`.

N.B: There is an analytical solution available for this linear case.

We can also add a constant bias term for both users and movies. We can also use a sigmoid to clamp the values at the end. Sigmoid seems to work better than a simple `torch.clamp` function.

#### Neural Net Collaborative Filtering

We can concatenate these movie / user embeddings into a single vector, and then feed that vector into a fully connected neural network.
