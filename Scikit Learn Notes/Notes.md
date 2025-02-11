## What is scikit learn used for:

1. Fit and predict estimators
2. Transformers and pre processors
3. Pipelines
4. Model Evaluation
5. Automated parameter search(mostly hyperparameter tuning)


## scikit learn important commands:

### 1. import sklearn
import the sklearn library

### 2. model.fit() and model.predict()
Scikit learn has built in ML algorithms and models, called estimators.

Each estimator can be  fitted to some data using the fit method and can predict the result using predict method.
e.g

```from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(random_state=42)

#define X and y values based on your data
model.fit(X, y)

#for predicting
model.predict(X)
```

Supervised learning estimators:
- Linear Models 
    - Linear regression
    - Ridge regression
    - Lasso regression
    - ElasticNet regression
    - Logistic regression
    - Perceptron
- Naive Bayes
    - Gaussian Naive Bayes
    - Multinomial Naive Bayes
    - Bernoulli Naive Bayes
- Decision Trees
     - Decision tree classifier
     - Decision tree regressor
- Ensembles
    - Random forest classifier
    - Random forest regressor
    - Gradient boosting classifier
    - Gradient boosting regressor
    - AdaBoost classifier
    - AdaBoost regressor
- Support Vector Machines(SVMs)
    - SVM Classifier
    - SVM Regressor
- Nearest Neighbors
    - K Nearest Neighbors Classifier
    - K Nearest Neighbors Regressor
- Neural Networks
    - Multi Layer perceptron classifier
    - Multi Layer perceptron regressor

Supervised learning estimators:
- Clustering
    - K Means clustering
    - Agglomerative clustering
    - DBSCAN
    - Gaussian Mixture model
- Dimensionality Reduction
    - Principal component analysis(PCA)
    - Linear Discriminant analysis(LDA)
    - Non-negative Matrix factorization(NMF)
    - t distributed Stochastic Neighbor Embedding(t-SNE)

### 3. model.score[X, y]
This is for predicting accuracy of the prediction.

Accuracy is the percentage of observed prediction vs real class.

### 4. Preprocessing with StandardScaler
Pre-processors and transformers follow the same API as the estimator objects. 

Transformer objects dont have a predict method but rather a transform method that outputs a newly transformed sample matrix X.
e.g
```
from sklearn.preprocessing import StandardScaler

Fx = StandardScaler().fit(X).transform(X)

#or can also use

StandardScaler().fit_transform(X)

#Can we transform without using fit?
#Yes, we can fit on some other data, and then run transform - mean and variance come from fit. Train/test.

Y = np.array([1.1, 2.2, 4.4, 5.5, 6.6]).reshape(-1, 1)

Fx.transform(Y)
```

### 5. To apply different transformations to different features, ColumnTransformer is designed for these use cases.
```
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction import CountVectorizer
from sklearn.preprocessing import OneHotEncoder

column_trans = ColumnTransformer(...)

column_trans.fit(X)
```

### 6. Pipelines
Transformers(data preprocessing) and estimators(ML Algorithms) combined together into a single unifying object is called a pipeline.

Pipeline can be fitted and used for prediction with fit and predict.

Using a pipeline prevents from data leakage i.e disclosing some testing data with training data

 ```
 pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression()
 )

#All estimators in a pipeline except the last one must be transformers(i.e must have transform method)
#Last estimator may be of any type(transformer, classifier, etc)
 ```

### 7. Model Evaluation
sklearn also provides many tools for model evaluation, in particular for cross validation.

Fitting a model to some data does not entail that it will predict well on unseen data, this need to be evaluated.

```
X, y = make_regression(n_samples=100, random_state=0)

result = cross_validate(LinearRegression(), X, y) #default is 5 fold cv
```
### 8. Automated parameter search(Hyperparameter tuning)
All estimators have parameters(k/a hyperparameters) that can be tuned.

Generalization power of estimator depends on a few parameters. 

e.g RandomForestRegressor has n_estimators parameter that determines the number of trees in the forest, max_depth determined the maximum depth of each tree

It is not often clear what the exact values of these parameters should be, since they depend on the data at hand.

We have cross validations:
- GridSearchCV
- RandomizedSearchCV

