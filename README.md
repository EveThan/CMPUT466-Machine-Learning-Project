# CMPUT466 Machine Learning Final Project

## Introduction
In this project, 3 machine learning algorithms are implemented using the Scikit-learn library on a dataset about insurance beneficiaries’ personal information and the individual medical costs charged by the health insurance. The goal is to use the machine learning algorithms to predict the amount of medical costs charged by health insurance given the personal information of the beneficiaries and compare the performances of the algorithms. The 3 algorithms employed are
- Stochastic Gradient Descent Regression (SGD Regression) at <a href="https://github.com/ZhengEnThan/CMPUT466-Machine-Learning-Final-Project/blob/main/SGD_regression.py" target="_blank">SGD_regression.py</a>, 
- Support Vector Regression (SVR) at <a href="https://github.com/ZhengEnThan/CMPUT466-Machine-Learning-Final-Project/blob/main/SVR.py" target="_blank">SVR.py</a>, and
- Decision Tree Regression at <a href="https://github.com/ZhengEnThan/CMPUT466-Machine-Learning-Final-Project/blob/main/Decision_tree_regression.py" target="_blank">Decision_tree_regression.py</a>.

A trivial algorithm (<a href="https://www.google.com/" target="_blank">Trivial_baseline.py</a>) is also implemented to serve as a baseline for comparison. The algorithms are run on the same dataset separately and the mean squared errors of their predictions on the test set are compared with each other.

## Problem Formulation 
### Background of the Dataset
The dataset used in this project is the file <a href="https://www.google.com/" target="_blank">insurance.csv</a> which is downloaded from Kaggle. The dataset is originally from the book Machine Learning with R by Brett Lantz. The dataset can be downloaded at <a href="https://www.kaggle.com/mirichoi0218/insurance" target="_blank">Medical Cost Personal Datasets on Kaggle</a>.

### Number of Rows (Samples) and Columns
The dataset consists of 1338 rows which correspond to 1338 samples and there are 7 columns in total.

### Input of the Machine Learning Models
The input of the machine learning models are the first 6 columns of the dataset, which are
- Age: Age of primary beneficiary, which takes an integer value,
- Sex: Insurance contractor gender, which takes the strings “female” or “male”,
- Bmi: Body mass index of the insurance contractor in kg/m2, which takes a float number,
- Children: Number of children or dependents covered by the health insurance, which takes
an integer value,
- Smoker: Whether the insurance contractor smokes, which takes the strings “yes” or “no”,
and
- Region: The beneficiary’s residential area in the US, which takes the strings “northeast”,
“northwest”, “southwest”, or “southwest”.

### Output of the Machine Learning Models
The last 1 column in the dataset serves as the target or output of the machine learning models. The column is labelled “charges” and it consists of individual medical cost billed by health insurance. The entries of the column contain float numbers.

## Approaches and Baselines
The 3 machine learning algorithms implemented in this project are Stochastic Gradient Descent Regression (SGD Regression), Support Vector Regression (SVR), and Decision Tree Regression.

### Main Libraries for Algorithm Implementation
The 3 machine learning algorithms are implemented with the help of several Scikit-learn libraries, which are
- sklearn.linear_model.SGDRegressor for SGD Regression,
- sklearn.svm.SVR for SVR, and
- sklearn.tree.DecisionTreeRegressor for Decision Tree Regression.

### Train-Validation-Test Infrastructure and Hyperparameter Tuning
The train-validation-test infrastructure is employed in the 3 machine learning algorithms in several steps, which are
- splitting the dataset into training set and test set using sklearn.model_selection.train_test_split, where 20% of the data will be the test set,
- training the machine learning models on the training set and tuning the hyperparameters at the same time using sklearn.model_selection.RandomizedSearchCV, where K-Folds cross validation is used with the number of folds set to 3 and mean squared error is used as the scoring method to evaluate the performance of the cross-validated model, and
- using the trained and validated model to predict the output of the test set.

The K-Folds cross validation implemented here does not shuffle the dataset by default. RandomizedSearchCV is used instead of GridSearchCV because RandomizedSearchCV is much faster than GridSearchCV.

### Encoding Categorical Features
3 features of the dataset, which are “sex”, “smoker”, and “region” do not have numerical input. Therefore, sklearn.compose.ColumnTransformer and sklearn.preprocessing.OneHotEncoder are used to encode the categorical features as one-hot indicator vectors.

### Feature Scaling
Feature scaling is implemented in all 3 machine learning algorithms because it is recommended to improve the performance of the SVR model. In order to have the errors of all the algorithms to be in the same scale for easier comparison, feature scaling is implemented in all 3 of the machine learning algorithms. Feature scaling is employed by using the library sklearn.preprocessing.StandardScaler.

### Trivial Baseline
A trivial algorithm is also implemented to serve as a baseline in comparing the different machine learning models. The trivial algorithm is very similar to the other 3 algorithms in the first half of the code. Feature scaling is also employed in the trivial algorithm so that its result or error will be in the same scale as the other algorithms. The dataset is split into the training set and test set as well. However, the model is not trained or validated. Instead, it simply produces random numbers that are lower bounded and upper bounded by the minimum value and the maximum value of the training set output respectively.

## Evaluation Metric
The evaluation metric used in all 3 machine learning algorithms and the 1 trivial algorithm implemented in this project is the mean squared error of the predicted output. The predicted output of the test set is compared to the true output of the test set and the difference between their values is measured using mean squared error. The library used to find the mean squared error is sklearn.metrics.mean_squared_error.
Having a lower mean squared error indicates a better performance in predicting the test set output. This measure of success is in line with the real goal of the task because the real goal is to predict the output, which in this case is the amount of individual medical cost billed by health insurance, as correctly as possible.
  
## Results
The mean squared errors calculated for each of the algorithms are
- SGD Regression: 0.21724472287025137
- SVR: 0.13360922987458954
- Decision Tree Regression: 0.29796826880800303
- Trivial baseline: 6.200584893044658

It is obvious that SGD Regressor, SVR, and Decision Tree Regression all performed much better than the trivial algorithm because the trivial baseline has a much higher mean squared error. This implies that the 3 machine learning algorithms have learned some patterns in the training dataset and applied their learned models on the test set instead of just guessing the test set output randomly.

Among the 3 machine learning algorithms, we can also conclude that SVR performed the best among all algorithms. SGD Regression came off second best while Decision Tree Regression came last.

There could be several reasons why Decision Tree Regression is not performing as well as the other algorithms. The Decision Tree algorithm has been known to give lower prediction accuracy compared to other machine learning algorithms in general and is not that good at extrapolating outside of the data.

The SGD Regression algorithm employs a linear model. The relationship between the features and the output might not be truly linear which may be the reason why SGD Regression is not performing as well as SVR in this particular case.
SVR is known to be robust and acknowledges the presence of non-linearity in data. It is also in general highly accurate and has an excellent generalization capability. These advantages may be the reasons why SVR performed the best here.

## References
- Understanding Decision Tree, Algorithm, Drawbacks and Advantages. <br>
https://medium.com/@sagar.rawale3/understanding-decision-tree-algorithm-drawbacks-and-advantages-4486efa6b8c3
- Regression using decision tree <br>
https://stats.stackexchange.com/questions/532601/regression-using-decision-tree
- SVM Hyperparameter Tuning using GridSearchCV <br>
https://www.vebuso.com/2020/03/svm-hyperparameter-tuning-using-gridsearchcv/

~ Project created in December 2021 ~
