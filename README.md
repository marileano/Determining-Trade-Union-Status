# Determining-Trade-Union-Status

## Project Overview
In this project we built a model to predict whether a person will remain in a hypothetical trade union called the United Data Scientists Union (UDSU).  Assume a country with a major trade union for data scientists.  Each member pays some dues amount to it each month and gets various benefits such as representation with employers and continuing education. The project provides a dataset of 1,000 data scientist IDs.  Each ID represent a data scientist who is a current or former member. The dataet will contain several features such as:
- Whether the person is a member of the management
- Number of months the particular data scientist has been a member of the UDSU
- Financial dues paid (each month, and in total)
- Gender of the person
- Status (still a union member, or no longer a member)

The project can be split into three parts: **data cleaning**, **correlation and PCA**, and **feature selection and modeling**.

## Objective
- Utilizing a custom TRAIN dataset (provided as csv), a model was built to predict whether a data scientist will remain a UDSU member.
- Each dataset will contain both numerical and categorical features.  Also includes some outliers, some missing values, and some colinear features.
- The dataset must be cleaned and standardized.

## Data Cleaning
The first thing to do is check for NaN values in the training data.  For the training data given, there were no missing values.  Then, we can look at the feature types (float, int, object) to see if there are numerical features that are of type ‘object’.  This will indicate that the variables are strings.  Looking at the feature types, TotalDues is considered an object so that variable was converted to a float type.  After, resulting NaN values were replaced with the median value.
Third, looking at only MonthsInUnion, MonthlyDues and TotalDues (dropping Management), we want to look at outliers in continuous variables.  We notice that the 3rd quartiles of MonthlyDues and MonthsInUnion are way smaller than the maximum values.  This might suggest the existence of outliers.  To check this, we can create a boxplot.
<p align="center">
  <img src="https://user-images.githubusercontent.com/58959101/157134953-0c6ba4d7-97cd-4a2d-bea9-d802e1d0bd21.png" /p>
 </p>
 <p align="center">
  <img src="https://user-images.githubusercontent.com/58959101/157134977-00e51625-8860-4050-8e9f-71b8434ad992.png" /p>
 </p>
 
Based on the boxplot, there are indeed outliers that impair our models.  In this case we should remove data less than 2000 in the MonthlyDues and less than 500 in MonthsInUnion.
Lastly, we can perform the conversion to numeric and standardization.  This consists of using the function LabelEncoder to perform encoding for binary variables.  As for the multiclass ones we can use One-Hot Encoding.  We can also perform get_dummies for categorical variables and StandardScaler to standardize the float variables.
 <p align="center">
  <img src="https://user-images.githubusercontent.com/58959101/157135582-e8378db0-c8c9-412f-bd09-db39aee42e3f.png" /p>
 </p>
 Then our data will look like this after cleaning up.
 <p align="center">
  <img src="https://user-images.githubusercontent.com/58959101/157135661-785e808e-3946-493c-8e41-21b991eda74c.png" /p>
 </p>
 
## PCA and Correlation
In this section, we will first plot scatter plots for continuous variables.  Plotting scatter plots allows us to see if there are any patterns between the three continuous variables.  Based on the scatter plot below, there are no obvious patterns.
In this section, we will first plot scatter plots for continuous variables.  Plotting scatter plots allows us to see if there are any patterns between the three continuous variables.  Based on the scatter plot below, there are no obvious patterns.
 <p align="center">
  <img src="https://user-images.githubusercontent.com/58959101/157135994-7dbf25c6-b5e0-48f0-a8aa-ffd75b1257ff.png" /p>
 </p>

Then, we will calculate the Pearson correlation coefficient between the variables, the target, as well as the p-values.  Finding the correlation coefficients allows us to find how strong a relationship is between data.  The output will return a value between -1 and 1 where: 1 indicates a strong positive relationship, -1 indicates a strong negative relationship and 0 indicates no relationship at all.  From the Pearson correlation matrix, we can see that there is not a single variable that is highly correlated with the target.  Later we will perform Recursive Feature Elimination with the modeling section so that we can consider smaller sets of features.
 <p align="center">
  <img src="https://user-images.githubusercontent.com/58959101/157136029-b6764d44-fc02-4c60-a4cb-596ae3a0a91b.png" /p>
 </p>

Finally, we will apply PCA to the data and plot the Scree Plot.  The scree plot shows us that most of the variance can be explained with the two first principal components.  We tried training the models with them (and adding more later), but it gave us results that were either less or as much as the features.  In this case it was better to just go with the current features and perform feature selection.
 <p align="center">
  <img src="https://user-images.githubusercontent.com/58959101/157136064-27c3d1fe-e9a5-4e5f-9ba3-8b88cce580bc.png" /p>
 </p>

## Feature Selection and Modeling
We will use four models in our modeling phase: Logistic Regression, SVM with a polynomial kernel, Decision Tree and Random Forest.  To select features, we will use Recursive Feature Elimination (RFE), which is a wrapper-type feature selection algorithm.  RFE starts with all the features in the dataset and fitting a ML algorithm, then ranking features by importance, removing the least important features and re-fitting the model until it reaches a desired number.
We will use RFE with Logistic Regression to extract optimal features, then we will use them to fit the same model, as well as SVM.  We will use it again with Decision Tree to extract other optimal features and then fit it.  And finally do the same for Random Forest.
We will also optimize the models by the following:
  -	Finding the best polynomial degree for the SVM
  -	Finding the maximum depth for Decision Tree
  -	Finding the maximum depth for Random Forest
We find that the best model in terms of training convergence as well as testing accuracy is the **Random Forest Model** (even though SVM had a higher testing accuracy, it had less training accuracy, which sets it behind).
 <p align="center">
  <img src="https://user-images.githubusercontent.com/58959101/157136777-7c3b5b9d-622f-4aad-ad82-4ca7cd4f9c0a.png" /p>
 </p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/58959101/157136800-51c5c337-5d2a-4989-a6b7-a72bbd975e53.png" /p>
 </p>
With the Random Forest Model, we can also print its classification report.
Lastly, we use the Random Forest to predict the final testing dataset.  In the Predict and Save section we basically call the test data and run our best model to get the output of our results which will be saved in a csv for a final submission.
