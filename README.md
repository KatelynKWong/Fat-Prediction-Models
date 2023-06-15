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

Next, I for-looped through the remaining feature columns and fit a LinearRegression model on each column against the total fat variable. Similar to the exploratory data analysis process, this showed me the correlation relationship between each feature with total fat and further narrowed my final choice of features to use in my linear regression model.
***show output dictionary***

I then decided to use the top three features that had the highest r^2 linear regression score in my baseline model: `calories`, `saturated_fat` and `protein`. All of these features are continuous quantitative features, which I believed would help accurately predict a recipe's total fat content using regression. I ended up using each feature's original values through a pipeline that preserves the original data without encoding or transforming the data; the main reason is that when looking at the regression scatterplots, the data values seemed to follow a linear regression pattern with relatively evenly spread out residuals.

Overall, the model seemed to perform satisfactorily. I evaluated it using the RMSE and r^2 values for both the training and testing data sets. The RMSE values I calculated for the training and testing data sets were `23.05` and `22.28` respectively. The r^2 values I calculated for the training and testing data sets were `0.8228` and `0.8487`. The r^2 values show that there is a moderately strong linear correlation between the features `calories`, `saturated_fat` and `protein` that I included in my model and the response variable `total_fat`.

---
## Final Model

For my final model, I wanted to improve on the prediction accuracy of my base model. 

Here, I decided to add `average_rating` as a predictive feature to my model because in Project 3, I found that there was a statistically signficant correlation between `average_rating` and `total_fat`. However, I decided to binarize the `average_ratings` column because the distribution of ratings was left skewed with the majority of the ratings being 5 (shown in the graph below). So I chose to set a threshold level of 4 on the Binarizer to split the ratings data into higher rated recipes (4, 5) and lower rated recipes (1, 2, 3)
*** insert graph of distribution of average_ratings***

I also decided to add `carbs` as a feature to my model. When performing the LinearRegression on `carbs` and `total_fat` earlier, the r^2 value revealed how there was a positive linear correlation between `carbs` and `total_fat`. However, since the r^2 values for both `protein` and `carbs` were below 0.3, I decided to transform the feature columns using std_scaler to normalize the values and eliminate any data redundancy and inconsistent dependency. For the features `calories` and `saturated_fat`, I chose to leave the values untransformed so that the regression model would receive the raw data since both are strongly positively correlated with `total_fat`.

I then selected the DecisionTreeRegressor model algorithm as my main predicting model because the relationship between some of the features was more complex than a simple regression model and the DecisionTreeRegressor model would help break down the data into more manageable parts using a series of questions on the dataset. To find the best combination of hyperparameters for my DecisionTreeRegressor model, I used used the GridSearchCV to loop through different combinations of hyperparameters and perform a k-fold cross validation on each combination. Below is my dictionary of hyperparameters that I looped through and I chose to use a 9-fold cross validation for each hyperparameter combination. 
*** insert dictionary of hyperparameters ***

Although there is some variability in best hyperparameters when conducting the GridSearchCv on different training sets, generally, the best hyperparameters chosen for the DecisionTreeRegressor model were a squared_error `criterion`, a `max_depth` of 22, and a `min_samples_split` value of 2. I then saved these chosen best hyperparameters into 3 variables.


When evaluating the performance of my DecisionTreeRegressor model against my baseline LinearRegression model, there is a drastic increase in r^2 values and decrease in RMSE values from the LinearRegression model to the DecisionTreeRegressor model. The RMSE values for the training and testing data sets were 0.273 and 9.83 respectively. The r^2 values for the training and testing data sets were **0.9999** and **0.9675** respectively. This reveals how the final model's error values are lower than the baseline model and prediction accuracy score is much higher than the baseline model. Ultimately, we can see that there is a significant improvement in model performance.

Note:
    I also experimented with a KNNeighbors prediction model and a more advanced version of the Linear Regression model. For both of these algorithms, I transformed the same feature columns and fit them into both models. For the KNNeighbors model, I also performed a search for the best `n_neighbors` hyperparameter value and used this in the final KNNeighbors model. However, the RMSE (15.35) and r^2 (0.9208) values of the testing dataset for the KNNeighbors model revealed that the predictive performance of this model is not as accurate as the DecisionTreeRegressor model. I also decided not to use the Advanced LinearRegression model, which was essentially a normal LinearRegression on the same transformed input feature columns. However, depending on how the training data was split, the RMSE and r^2 values of this model were usually similar to the DecisionTreeRegressor model so I decided to use the DecisionTreeRegressor model as my final predicting model.

In summary, the overall final DecisionTreeRegressor model consisted of
1. the following feature column transformations:
    - Binarization of `average_rating` on a threshold of 4 
    - Scaling `protein` and `carbs`
2. keeping the `calories` and `saturated_fat` columns untransformed,
3. Finding the best hyperparameters for the DecisionTreeRegressor algorithm using a GridSearchCV algorithm,
4. and fitting the final input feature columns of the training data into the DecisionTreeRegressor algorithm with the best hyperparameter combination from the GridSearchCV algorithm.

---
## Fairness Analysis

For my fairness analysis, I tested my Final model on the two groups: recipes with `average_rating` of at least 4 and recipes with `average_ratings` of less than 4. Because my model is a regression that uses only quantitative features, my fairness analysis will utilize the r^2 accuracy value as the evaluation metric. 

I then performed a permutation test, using the binarized `average_rating` column to evaluate whether the model performs fairly across both higher rated and lower rated recipe groups. 

- Null hypothesis: The r^2 score of the DecisionTreeRegressor model is the same for recipes with average ratings of at least 4 and recipes with average ratings of less than 4.
- Alternative hypothesis: The r^2 score of the DecisionTreeRegressor model is higher for recipes with average ratings of at  least 4 than recipes with average ratings of less than 4.
- test statistic: signed difference in r^2 scores
- significance level: 0.01

After performing the permutation test, I obtained a p-value of __ . Since the value is much higher than our significance level of 0.01, we fail to reject the null hypothesis an conclude that there is not enough statistical evidence of a significant difference between the r^2 values of both groups. Ultimately, there is likely no difference between the DecisionTreeRegression model's r^2 scores for recipes with average ratings less than 4 and average ratings greater than 4. Therefore, my final model appears to perform fairly when the binary attribute of interest is whether or not the average rating of a recipe is greater than 4.
