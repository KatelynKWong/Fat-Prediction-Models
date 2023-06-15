# Can we predict the total fat content in a recipe using nutrition facts and recipe ratings?
by Katelyn Wong (kkw004@ucsd.edu)

---
This project contains data from DSC80 Project 3. 
The exploratory data analysis can be found [on this website](https://katelynkwong.github.io/Food.com-Recipe-Analysis/).

After initial exploratory data analysis and testing of the dataset of recipe information from food.com from Project 3, I decided to dive deeper into machine learning prediction models to predict a feature of a recipe using given input features. This site documents the process of developing and testing my prediction model.

---
## Framing the Problem

The prediction problem I ultimately wanted to solve was to **predict how much total fat is in each recipe** given the features of each recipe. This is a regression prediction problem with the response variable being the total fat (PDV) of a recipe.

I chose to use total fat (PDV) as my response variable because I wanted to continue my exploration of the relationship between fat and other features of a recipe. Because I had already established in Project 3 that there is a statistically significant relationship between a recipe's average rating and fat content, I wanted to explore further by constructing a prediction model and including average ratings to predict a recipe's fat content. In addition, I was more interested in the relationship of total fat with other features rather than saturated fat; as such, I narrowed down my desired response variable to total fat.

I decided to primarily use accuracy values to evaluate my model, focusing on both the r^2 score and RMSE score, due to the regressive nature of the problem. I wanted to include both metrics in my model evaluation because the r^2 value measures the correlation between the response variable and the independent variable while the RMSE gives higher weight to large error, allowing us to detect larger errors.

---
## Baseline Model

For the baseline model, I decided to use a simple linear regression using 3 quantitative features to predict the total fat content in a recipe. 

First, I cleaned out the data by dropping any columns containing features that likely would not help predict total fat: These features were `recipe_id`, `date_submitted`, `interaction_date`, `name`, `indiv_rating`. 

|    |   average_rating |   minutes |   n_steps |   n_ingredients |   calories |   total_fat |   sugar |   sodium |   protein |   saturated_fat |   carbs |
|---:|-----------------:|----------:|----------:|----------------:|-----------:|------------:|--------:|---------:|----------:|----------------:|--------:|
|  0 |                4 |        40 |        10 |               9 |      138.4 |          10 |      50 |        3 |         3 |              19 |       6 |
|  1 |                5 |        45 |        12 |              11 |      595.1 |          46 |     211 |       22 |        13 |              51 |      26 |
|  2 |                5 |        40 |         6 |               9 |      194.8 |          20 |       6 |       32 |        22 |              36 |       3 |
|  3 |                5 |        40 |         6 |               9 |      194.8 |          20 |       6 |       32 |        22 |              36 |       3 |
|  4 |                5 |        40 |         6 |               9 |      194.8 |          20 |       6 |       32 |        22 |              36 |       3 |


Next, I for-looped through the remaining feature columns and fit a LinearRegression model on each column against the total fat variable. Similar to the exploratory data analysis process, this reveals the correlation relationship of each feature against total fat and further narrowed my final choice of features to use in my linear regression model.

```python
>>> print(r_squared_dict)
{'average_rating': 8.720937407113993e-06,
 'minutes': 1.1371655497249833e-06,
 'n_steps': 0.017626350696544058,
 'n_ingredients': 0.013581414394469582,
 'calories': 0.7563815323956382,
 'sugar': 0.16263160922331887,
 'sodium': 0.015227248462355014,
 'protein': 0.26022738746402185,
 'saturated_fat': 0.7436474690764439,
 'carbs': 0.21142279129203156}
```


I chose the top three features that had the highest r^2 linear regression score in my baseline model: `calories`, `saturated_fat` and `protein`. All of these features are continuous quantitative features, which I believed would help accurately predict a recipe's total fat content using regression. When looking at the regression scatterplots, the data values seemed to follow a linear regression pattern with relatively evenly spread out residuals pattern. Therefore, I used each feature column's raw values by inputting them through a pipeline that preserves the original data without encoding or transforming the data

<iframe src="assets/cal_v_fat.html" width=800 height=600 frameBorder=0></iframe>
<iframe src="assets/sfat_v_fat.html" width=800 height=600 frameBorder=0></iframe>
<iframe src="assets/prot_v_fat.html" width=800 height=600 frameBorder=0></iframe>

Overall, the model seemed to perform passably. I evaluated its performance using the RMSE and r^2 values for both the training and testing data sets. The RMSE values I calculated for the training and testing data sets were `22.955` and `22.235` respectively; the r^2 values I calculated for the training and testing data sets were `0.8312` and `0.8298`. The r^2 values provide evidence that there is a moderately strong linear correlation between the features, `calories`, `saturated_fat` and `protein`, that I included in my model and the response variable `total_fat`.

---
## Final Model

For my final model, I wanted to improve on the prediction accuracy of my base model. 

Here, I decided to add `average_rating` as a predictive feature to my model because in Project 3, I found that there was a statistically signficant correlation between `average_rating` and `total_fat` (see graph below). However, I decided to binarize the `average_ratings` column because the distribution of ratings was left skewed with the majority of the ratings being 5. I therefore set a threshold level of 4 on the Binarizer to split the ratings data into higher rated recipes (4-5) and lower rated recipes (1-3).

<iframe src="assets/arating_v_fat.html" width=800 height=600 frameBorder=0></iframe>

I also decided to add `carbs` as a feature to my model. When performing the LinearRegression on `carbs` and `total_fat` earlier, the r^2 value revealed how there was a positive linear correlation between `carbs` and `total_fat` (see `r_squared_dict` above). However, since the r^2 values for both `protein` and `carbs` were below 0.3, I decided to transform the feature columns using Sklearn's StandardScaler to normalize the values and eliminate any data redundancy and inconsistent dependency. For the features `calories` and `saturated_fat`, I chose to leave the values untransformed so that the regression model would receive the raw data of both feature columns, which were strongly positively correlated with `total_fat`.

```python
preproc = ColumnTransformer([
    ('ratings_quant', Binarizer(threshold = 4), ['average_rating']), # Binarizes `average_rating` into 1s and 0s
    ('std_scaler', scalepipe, ['protein', 'carbs']), # applies StandardScaler() to `protein` and `carbs`
    ('keep_class', keeppipe, ['calories','saturated_fat']) # uses raw data 
])
```

I then selected the DecisionTreeRegressor model algorithm as my main predicting model because the relationships between the features were more complex and the DecisionTreeRegressor model would help break down the data into more manageable parts using a series of questions on the dataset. To find the best combination of hyperparameters for my DecisionTreeRegressor model, I used used Sklearn's GridSearchCV to loop through different combinations of hyperparameters and perform a 9-fold cross validation on each combination. Below is my dictionary of hyperparameters that I looped through:

```python
>>> print(hyperparameters)
{'max_depth': [13, 15, 18, 22, 25, 27, None], 
 'min_samples_split': [2, 4, 5, 6,7],
 'criterion': ['squared_error','friedman_mse']}
```

Although there is some variability in best hyperparameters when conducting the GridSearchCv on different training sets, generally, the best hyperparameters chosen for the DecisionTreeRegressor model were a friedman_mse `criterion`, a `max_depth` of 18, and a `min_samples_split` value of 4. I then saved these chosen best hyperparameters into 3 variables.


When evaluating the performance of my DecisionTreeRegressor model against my baseline LinearRegression model, there is a drastic increase in r^2 values and decrease in RMSE values from the LinearRegression model to the DecisionTreeRegressor model. The RMSE values for the training and testing data sets were 1.435 and 8.513 respectively. The r^2 values for the training and testing data sets were **0.9993** and **0.9751** respectively. This reveals how the final model's error values are lower than the baseline model and prediction accuracy score is much higher than the baseline model. Ultimately, we can see that there is significant improvement in model performance.

Note:
    I also experimented with a KNNeighbors prediction model and a more advanced version of the Linear Regression model. For both of these algorithms, I transformed the same feature columns and fit them into both models. For the KNNeighbors model, I also performed a search for the best `n_neighbors` hyperparameter value and used this in the final KNNeighbors model. However, the RMSE (15.271) and r^2 (0.9197) values of the testing dataset for the KNNeighbors model revealed that the predictive performance of this model is not as accurate as the DecisionTreeRegressor model for this data set. I also decided not to use the Advanced LinearRegression model, which was essentially a normal LinearRegression on the same transformed input feature columns. Even though the RMSE and r^2 values of this model were similar to the DecisionTreeRegressor model depending on how the training data was split, I still chose the DecisionTreeRegressor model as my final predicting model.
    
|    | model                    |      r^2 |     RMSE |
|---:|:-------------------------|---------:|---------:|
|  0 | base                     | 0.829823 | 22.2345  |
|  1 | decision tree regression | 0.975056 |  8.51263 |
|  2 | knn                      | 0.919728 | 15.2708  |
|  3 | linear regression        | 0.98565  |  6.45653 |


In summary, the overall final DecisionTreeRegressor model consisted of
1. the following feature column transformations:
    - Binarization of `average_rating` on a threshold of 4 
    - Scaling `protein` and `carbs`
2. maintaining the raw, untransformed data in the `calories` and `saturated_fat` columns,
3. finding the best hyperparameters for the DecisionTreeRegressor algorithm using a GridSearchCV algorithm,
4. and fitting the final input feature columns of the training data into the DecisionTreeRegressor algorithm with the best hyperparameter combination from the GridSearchCV algorithm.

```python
>>> print(dtr_pipe)
Pipeline(steps=[('preprocessor',
                 ColumnTransformer(transformers=[('ratings_quant',
                                                  Binarizer(threshold=4),
                                                  ['average_rating']),
                                                 ('std_scaler',
                                                  Pipeline(steps=[('scale',
                                                                   StandardScaler())]),
                                                  ['protein', 'carbs']),
                                                 ('keep_class',
                                                  Pipeline(steps=[('keep',
                                                                   FunctionTransformer(func=<function <lambda> at 0x0000026AB033CF70>))]),
                                                  ['calories',
                                                   'saturated_fat'])])),
                ('regression',
                 DecisionTreeRegressor(criterion='friedman_mse', max_depth=18,
                                       min_samples_split=4))])
```
---
## Fairness Analysis

For my fairness analysis, I tested my Final model on two groups: recipes with `average_rating` of at least 4 and recipes with `average_ratings` of less than 4. Because my model is a regression that uses only quantitative features, my fairness analysis will utilize the r^2 accuracy value as the evaluation metric. 

I then performed a permutation test by binarizing the `average_rating` column and using this new feature column (`binarized_rating`) to evaluate whether the model performs fairly across both higher rated and lower rated recipe groups.

|    |   average_rating |   minutes |   n_steps |   n_ingredients |   calories |   sugar |   sodium |   protein |   saturated_fat |   carbs |   total_fat | binarized_rating   |
|---:|-----------------:|----------:|----------:|----------------:|-----------:|--------:|---------:|----------:|----------------:|--------:|------------:|:-------------------|
|  0 |          4       |        15 |         8 |               8 |      306.9 |      15 |       11 |       105 |               6 |       2 |           9 | True               |
|  1 |          5       |        65 |        10 |               8 |       87   |       8 |        1 |         2 |              17 |       2 |           9 | True               |
|  2 |          3       |        65 |         8 |              13 |      643.5 |      40 |       40 |        63 |              53 |      18 |          52 | False              |
|  3 |          4.22222 |        45 |         9 |               6 |      275.1 |      48 |       20 |        69 |               8 |       4 |          12 | True               |
|  4 |          5       |        40 |         4 |               7 |      155.7 |      65 |       20 |        32 |               2 |       6 |           1 | True               |


- **Null hypothesis:** The r^2 score of the DecisionTreeRegressor model is the same for recipes with average ratings of at least 4 and recipes with average ratings of less than 4.
- **Alternative hypothesis:** The r^2 score of the DecisionTreeRegressor model is higher for recipes with average ratings of at  least 4 than recipes with average ratings of less than 4.
- **Test statistic:** signed difference in r^2 scores
- **Significance level:** 0.01

After performing the permutation test, I obtained a p-value of **0.59**. Since the value is much higher than our significance level of 0.01, we fail to reject the null hypothesis an conclude that there is not enough statistical evidence of a significant difference between the r^2 values of both groups. Ultimately, there is likely no difference between the DecisionTreeRegression model's r^2 scores for recipes with average ratings less than 4 and average ratings greater than 4. 

Therefore, my final model appears to perform fairly when the binary attribute of interest is whether or not the average rating of a recipe is greater than 4.
