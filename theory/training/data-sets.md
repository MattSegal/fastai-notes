# Training Data Sets

In general we want our models to perform well on new, unseen data, not just on training data. The way we choose and segment our data sets for traning have an important influence on this performance.

Use 3 data sets

- train
- validation
- test

The training set is used to train the model. Labelled data from the training set provides the error signal that allows the model to update.

The validation set is used to choose between a group of models. We may train both a neural net and a random forest on the training set, and then use the validation set to choose between them. We can also use the validation set to track the progress of overfitting, online. In a sense we are still using this data to choose two models - two versions of the same ANN that have done different amounts of training.

The test set is used to score the model, offline, according to some metric. We can also use it to reason about the behaviour of the model: does it tend to have good precision (true predictions are probably true)? Does it have good recall (false predictions are probably false)?

We want both the validation and test set to be representative of new data that we will see in the future. In some cases, simply taking a random subset of all the data we have available is OK, but in other cases that's not good enough. We have to be careful when we're dealing with time series data and qualitatively different data

#### Time Series Data

Splitting time series data up randomly by date will not work. Your model will simply learn to interpolate the gaps in the training set, and this will show up as a good score on the validation and test set. Instead, you want to simulate the conditions that the model will experience in reality: it knows everything about the past, but nothing about the future.

To achieve this you need to validate forecasts on non-randomized test/validation splits of the same time series. You want to choose a continuous section with the _latest_ dates as your validation set. Not sure how to extract a test set also.

#### Qualitatively Different Data

The data you make predictions on may be qualitatively different from your test data. For example, you may want to classify human poses in an image. You may have 1k images of 50 people. At prediction time, you are going to be classifying the pose of a _previously unseen_ person - someone who was not in the training set. To capture this requirement, we want to ensure that there are people in the validation / test set who are not in the training set.

### Cross Validation

Cross validation (CV) is useful for choosing one model from a group of candidate models. It is particularly useful for _small_ datasets, where you want to reduce the risk of overfitting your model to the validation set by twiddling hyperparameters. The idea is that you have a group of candidate models that you want to test (or a single model with different hyperparameters), and you want to figure out which model to train in anger. This technique is also known as "rotation estimation" or "out-of-sample testing".

In cross validation we use k different subsets of the training set for validation. In 3-fold (k=3) cross validation, the data is divided into 3 sets: A, B and C. You then:

- train on A + B, validate on C
- train on A + C, validate on B
- train on B + C, validate on A

And you use the average validation score as the model performance. The winning model is then trained on the entire training set (minus the final holdout test set).

Not everyone thinks that using CV is a good idea. If your intitial hyperparameters are "good enough", then there might not be a need for hyperparameter optimization and the CV required. Weight regularization techniques also offets the need for CV, becuase they reduce the risk of overfitting.

There's also nested cross validation, which is apparently useful for time series data. There's also forward chaining, which is better than nested cross validation for time series prediction... apparently.

### Data Leakage

Data leakage is when information from outside the training dataset is used to create the model. In particular, a feature that is not going to be available to model at prediction time is used in training. This is bad because it invalidates our evaluation of the model's performance. This can happen when:

- The test set 'leaks' into the training set
- The ground truth 'leaks' into the training data
- Information 'leaks' from the future into the past

There are two places where data can leak into - the feature variables, and the training set.

A trivial example is when the training set contains the answer - eg. you are predicting the weather, and one of the input features is the weather, so you model learns a tautology: "it is sunny on sunny days".

A more nuanced example is where we are working at a company and we want to maximise sales to existing customers. We feed in the customer's sales agent ID into the model as a feature. Here we have assumed that the identity of the sales agent is an independent variable:

- sales agent causes customer sales
- customer sales does not cause sales agent

We might be able to predict that some sales agents cause more sales than others, allowing us to optimize sales. What if our assumptions of independence weren't true though? What if customers with low sales, "problem cases", were all assigned to a particular sales agent? In that case, we will be smuggling the outcome back into the training data, and our predictions on new customers will not be useful.

Apparently including fields that tell the model what time it is in a timeseries prediction task are also at risk of leaking data into your training set... somehow.


Sources:

- [fast.ai](https://www.fast.ai/2017/11/13/validation-sets/)
- [insidebigdata.com](https://insidebigdata.com/2014/11/26/ask-data-scientist-data-leakage/)
