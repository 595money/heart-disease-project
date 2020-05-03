# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent,md
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Predicting heart disease using machine learning
# This notebook looks into using various Python-based machine learning and data science libraries in an attempt to build a machine learning model capable of predicting whether or not someone has heart disease based on their medical attributes.
#
# We're going to take the following approach:
# 1. Problem definition
# 2. Data
# 3. Evaluation
# 4. Features
# 5. Modelling
# 6. Experimentation

# %%
from jupyterthemes import jtplot
from IPython.core.display import display, HTML

display(HTML("<style>.container { width:80% !important; }</style>"))
jtplot.style()

# %% [markdown]
# ## 1. Problem Definition
# In a statement,
# > Given clinical parameters about a patient, can we predict whether or not they have heart disease?  
# 給定有關患者的臨床參數，我們能否預測他們是否患有心髒病？
# ## 2. Data  
# The original data came from the Cleavland data from the UCI Machine Learning Repository.  
# https://archive.ics.uci.edu/ml/datasets/Heart+Disease  
#   
# THere is also a version of it available on Kaggoe.  
# https://www.kaggle.com/ronitf/heart-disease-uci/data
#   
# ## 3. Evaluation  
# > If we can reach 95% accuracy at predicting whether or not a patient has heart disease during the proof of concept, we'll pursue the project.
# ## 4. Features  
# This is where you'll get different information about each of the features in your data.  
# **Create data dictionary**  
# 1. age - age in years
# 2. sex - (1 = male; 0 = female)
# 3. cp - chest pain type
#    * 0: Typical angina: chest pain related decrease blood supply to the heart
#    * 1: Atypical angina: chest pain not related to heart
#    * 2: Non-anginal pai:typically esophageal spasms (non heart realted)
#    * 3: Asymptomati: chest pain not showing signs of disease
# 4. trestbps - resting blood pressure (in mm Hg on admission to the hospital) anything above 130-140 is typically cause for concern  
# 5. chol - serum cholestoral in mg/dl
#    * serum = LDL + HDL + .2 * triglycerides
#    * above 200 is cause for concern
# 6. fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)  
#    * '>126' mg/dL signals diabets
# 7. restecg - resting electrocardiographic results
#    * 0: Nothing to note
#    * 1: ST-T Wave abnormality
# 8. thalach - maximum heart rate achieved
# 9. exang - exercise induced angina (1 = yes; 0 = no)
# 10. oldpeak - ST depression induced by exercise relative to rest looks at stress of heart stress more  
# 11. slope - the slope of the peak exercise ST segment
#    * 0: Upsloping: better heart rate with excercise (uncommon)
#    * 1: Flatsloping: minimal change (typical healthy heart)
#    * 2: Downslopins: signs of unhealthy heart
# 12. ca - number of major vessels (0-3) colored by flourosopy
#    * colored vessel means the doctor can see the blood passing through
#    * the more blood movement the better (no clots)
# 13. thal - thlium stress result
#    * 3: normal
#    * 6: fixed defect: used to bo defect but ok now
#    * 7: reversable defect: no proper blood movement when excercising 
# 14. target - jabe disease or not (1=yes, 0=no) (=the predicted attribute)  
#

# %% [markdown]
# ## Preparing the tools
# We're going to use pandas, Matplotlib and NumPy for data analysis and manipulation.

# %%
# Import all the tools we need


# Regular EDA (exploratory data analysis ) and plotting libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# We want our plots to appear inside the notebook
# %matplotlib inline

# Models from Scikit-Learn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# Model Evaluations
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import plot_roc_curve

# %% [markdown]
# ## Load data

# %%
df = pd.read_csv('source/heart.csv')
# (rows, columns)
df.shape

# %% [markdown]
# ## Data Exploration (exploratory data analysis or EDA)
# The goal here is to find out more about the data and become a subject matter export on the dataset you're working with.  
# 此處的目標是查找有關數據的更多信息，並成為正在使用的數據集上的主題導出。
#
# 1. What question(s) are you trying to solve?
# 2. What kind of data do we have and how do we treat different types?
# 3. What's missing from the data and how do you deal with it?
# 4. Where are the outliers and why should you care about them?
# 5. How can you add, change or remove features to get more out of your data?

# %%
# Let's find out how many of each class there
df['target'].value_counts()

# %%
df['target'].value_counts().plot(kind='bar', color=['salmon', 'lightblue']);

# %%
# non-null, 所以沒有 missing data
df.info()

# %%
# Are there any missing values?
# 也可以計算缺失值數目
df.isna().sum()

# %%
df.describe()

# %% [markdown]
# ### Heart Disease Frequency according to Sex

# %%
df.sex.value_counts()

# %%
# Compare target column with sex column
pd.crosstab(df.target, df.sex)

# %%
# Create a plot Diseaseof crosstab
pd.crosstab(df.target, df.sex).plot(kind='bar',
                                    figsize=(10, 6),
                                    color=['salmon', 'lightblue'])
plt.title('Heart Disease Frequency for Sex')
plt.xlabel('0 = No Diesease, 1 = Disease')
plt.ylabel('Amount')
plt.legend(['Female', 'Male'])
plt.xticks(rotation=0);

# %% [markdown]
# ### Age vs. Max Heart Rate for Heart Disease

# %%
# Create another figure
plt.figure(figsize=(10, 6));

# Scatter with postivie examples (確診病患 target == 1)
plt.scatter(df.age[df.target == 1],
            df.thalach[df.target == 1],
            c='salmon');

# Scatter with negitive examples (target == 0)
plt.scatter(df.age[df.target == 0],
            df.thalach[df.target == 0],
            c='lightblue');

# Add some helpful info
plt.title('Heart Disease in function of age and Max Heart Rate')
plt.xlabel('Age')
plt.ylabel('Max Heart Rate')
plt.legend(['Disease', 'No Disease']);

# %%
# Check the distribution of the age column with a histogram
df.age.plot.hist();

# %% [markdown]
# ### Heart Disease Frequency per Chest Pain Type
#
# 3. cp - chest pain type
#    * 0: Typical angina: chest pain related decrease blood supply to the heart
#    * 1: Atypical angina: chest pain not related to heart
#    * 2: Non-anginal pai:typically esophageal spasms (non heart realted)
#    * 3: Asymptomati: chest pain not showing signs of disease

# %%
pd.crosstab(df.cp, df.target)

# %%
# Make the crosstab more visual
pd.crosstab(df.cp, df.target).plot(kind='bar',
                                   figsize=(10, 6),
                                   color=['salmon', 'lightblue'])
# Add some communication
plt.title('Heart Disease Frequency Per Chest Pain Type')
plt.xlabel('Chest Pain Type')
plt.ylabel('Amount')
plt.legend(['No Disease', 'Disease'])
plt.xticks(rotation=0)

# %%
# Make a correlation matrix
# 檢視個欄位之間的相關性, 會排除掉 Na/null
df.corr()

# %%
# Let's make our correlation matrix a little prettier
# 特別關注正向指標 (越接近1)與反向指標(數字越小)
corr_matrix = df.corr()
fig, ax = plt.subplots(figsize=(15, 10))
ax = sns.heatmap(corr_matrix,
                 annot=True,
                 linewidths=0.5,
                 fmt='.2f',
                 cmap='YlGnBu')

# %% [markdown]
# ## 5.Modelling 

# %%
# Split data into X and y
X = df.drop('target', axis=1)
y = df['target']

# Split data into train and test sets
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2)

# %% [markdown]
# Now we've got our data split into training and test sets, it's time to build a machine learning model.  
# We'll train it (find the patterns) on the training set.  
# And we'll test it (use the patterns) on the test set.<br><br>
# We're going to try 3 different machine learning models:
# 1. Logistic Regression
# 2. K-Nearest Neighbours Classifier
# 3. Random Forest Classifier

# %%
# Put models in a dictionary
models = {'Logistic Regression': LogisticRegression(max_iter=10000),
          'KNN': KNeighborsClassifier(),
          'Random Forest': RandomForestClassifier()}

# Create a function to fit and score models
def fit_and_score(models, X_train, X_test, y_train, y_test):
    '''
    Fits and evaluates given machine learning models.
    models : a dict of different Scikit-Learn machine learning models
    X_train : training data (no labels)
    X_test : testing data (no labels)
    y_train : training labels
    y_test : test labels
    '''
    # Set random seed
    np.random.seed(42)
    
    # Make a dictionary to keep model scores
    model_scores = {}
    
    # Loop through models
    for name, model in models.items():
        
        # Fit the model to the data
        model.fit(X_train, y_train)
        
        # Evaluate the model and append its score to model_scores
        model_scores[name] = model.score(X_test, y_test)
    
    return model_scores


# %%
model_scores = fit_and_score(models=models,
                             X_train=X_train,
                             X_test=X_test,
                             y_train=y_train,
                             y_test=y_test)
model_scores

# %%
model_compare = pd.DataFrame(model_scores, index=['accuracy'])
model_compare.T.plot.bar();

# %% [markdown]
# Now we've got a baseline model... and we know a model's first predictions aren't always.  
# What we should based our next steps off. What should do?

# %% [markdown]
# Let's look at the following:
# * Hyperterparameter tuning (機器學習固定流程)  
# * Feature importance (機器學習固定流程)  
# * Confusion matrix
# * Cross-validation
# * Precision
# * Recall
# * F1 score
# * Classification report
# * ROC curve
# * Area under the curve (AUC)

# %% [markdown]
# ### Hyperparameter tuning (by hand)

# %%
# Let's tune KNN
train_scores = []
test_scores = []

# Create a list of different values for n_neighbors
neighbors = range(1, 21)

# Setup KNN instance
knn = KNeighborsClassifier()

# Loop through different n_neighbors
for i in neighbors:
    knn.set_params(n_neighbors=i)
    
    # Fit the algorithm
    knn.fit(X_train, y_train)
    
    # Update the training scores list
    train_scores.append(knn.score(X_train, y_train))
    
    # Update the test scores list 
    test_scores.append(knn.score(X_test, y_test))

# %%
train_scores

# %%
test_scores

# %%
plt.plot(neighbors, train_scores, label='Train score')
plt.plot(neighbors, test_scores, label='Test score')
plt.xlabel('Number of neighbors')
plt.xticks(np.arange(1, 21, 1))
plt.ylabel('Model score')
plt.legend()
print(f'Maximum KNN score on the test data: {max(test_scores) * 100:.2f}%')

# %% [markdown]
# ## Hyperparameter tuning with RandomizedSearchCV
#
# We're going to tune:
# * LogiticRegression()
# * RandomForestClassifier()  
#
# ... using RandomizedSearchCV

# %%
# CV > Cross-validation
# Create a hyperparameter grid for LogisticRegression
log_reg_grid = {'C': np.logspace(-4, 4, 20),
                'solver': ['liblinear']}

# Create a hyperparameter grid for RandomForestClassifier
rf_grid = {'n_estimators': np.arange(10, 1000, 50),
           'max_depth': [None, 3, 5, 10],
           'min_samples_split': np.arange(2, 20, 2),
           'min_samples_leaf': np.arange(1, 20, 2)}

# %% [markdown]
# Now we've got hyperparameter grids setup for each of our models,  
# let's tune them using RandomizedSearchCV...

# %%
# Tune LogisticRegression

np.random.seed(42)

# Setup random hyperparameter search for LogisticRegression
rs_log_reg = RandomizedSearchCV(LogisticRegression(),
                                param_distributions=log_reg_grid,
                                cv=5,
                                n_iter=20,
                                verbose=True)
# Fit random hyperparameter search model for LogisticRegression
rs_log_reg.fit(X_train, y_train)

# %%
rs_log_reg.best_params_

# %%
rs_log_reg.score(X_test, y_test)

# %% [markdown]
# Now we've tuned LogisticRegression(), let's do the same for RandomForestClassifier() ...

# %%
# Setup random seed
np.random.seed(42)

# Setup random hyperparameter search for RandomForestClassifier
rs_rf = RandomizedSearchCV(RandomForestClassifier(),
                           param_distributions=rf_grid,
                           cv=5,
                           n_iter=20,
                           verbose=True)
# Fit random hyperparameter search model for RandomForestClassifier()
rs_rf.fit(X_train, y_train)

# %%
# Find the best hyperparameters 
rs_rf.best_params_

# %%
# Evaluate the randomized search RandomForestClassifier model
rs_rf.score(X_test, y_test)

# %% [markdown]
# ## Hyperparameter Tuning with GridSearchCV  
# Since our LogisticRegression model provides the best scores so far,  
# we'll try and improve them again using GridSearchCV ...

# %%
# Different hyperparameters for our LogisticRegression model
log_reg_grid = {'C': np.logspace(-4, 4, 30),
                'solver': ['liblinear']}

# %%
# Setup grid hyperparameter search for LogisticRegression
gs_log_reg = GridSearchCV(LogisticRegression(),
                          param_grid=log_reg_grid,
                          cv=5,
                          verbose=True)

# Fit grid hyperparameter search model
gs_log_reg.fit(X_train, y_train)

# %%
# Check the best hyperparameters
gs_log_reg.best_params_

# %%
# Evaluate the grid search LogisticRegression model
gs_log_reg.score(X_test, y_test)

# %% [markdown]
# ## Evaluting our tuned machine learning classifier, beyond accuracy
# * ROC curve amd AUC score
# * Confusion matrix
# * Classification report
# * Precision
# * Recall
# * F1-score  
#
# ... and it would be great if cross-validation was used where possible.  
#
# To make comparisons and evaluate our trained model, first we need to make predictions.

# %%
# Make predictions with tuned model
y_preds = gs_log_reg.predict(X_test)

# %%
y_preds

# %%
# Plot ROC curve and calculate and calculate AUC metric
plot_roc_curve(gs_log_reg, X_test, y_test);

# %%
# Confusion matrix
print(confusion_matrix(y_test, y_preds))

# %%
sns.set(font_scale=1.5)
def plot_conf_mat(y_test, y_preds):
    '''
    Plots a nice looking confusion matrix using Seaborn's heatmap()
    '''
    fig, ax = plt.subplots(figsize=(3, 3))
    ax = sns.heatmap(confusion_matrix(y_test, y_preds),
                     annot=True,
                     cbar=False)
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    
plot_conf_mat(y_test, y_preds)

# %% [markdown]
# Now we've got a ROC curve, an AUC metric and a confusion matrix,  
# let's get a classification report as well as cross-validated precision, recall and f1-score.

# %%
# 此處的 y_test & y_preds 並未做交叉驗證所以不夠全面
print(classification_report(y_test, y_preds))

# %% [markdown]
# ### Calculate evaluation metrics using cross-validation
# We're going to calculate precision,  
# recall and f1-score of our model using cross-validation and to do so we'll be using `cross_val_score()`.

# %%
# Check best hyperparameters
gs_log_reg.best_params_

# %%
# Create a new classifier with best parameters
clf = LogisticRegression(C=0.20433597178569418,
                         solver='liblinear')

# %%
# Cross-validated accuracy
cv_acc = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring='accuracy')
cv_acc = np.mean(cv_acc)
cv_acc

# %%
# Cross-validated precision
cv_precision = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring='precision')
cv_precision = np.mean(cv_precision)
cv_precision

# %%
# Cross-validated recall
cv_recall = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring='recall')
cv_recall = np.mean(cv_recall)
cv_recall

# %%
# Cross-validated f1-score
cv_f1 = cross_val_score(clf,
                         X,
                         y,
                         cv=5,
                         scoring='f1')
cv_f1 = np.mean(cv_f1)
cv_f1

# %%
# Visualize cross- validated metrics
cv_metrics = pd.DataFrame({'Accuracy': cv_acc,
                           'Precision': cv_precision,
                           'Recall': cv_recall,
                           'F1': cv_f1},
                          index=[0])
cv_metrics.T.plot.bar(title='Cross-validated classification metrics',
                      legend=False);

# %% [markdown]
# ### Feature Importance  
# Feature importance is another as asking,  
# "Which features contributed most to the outcomes of the model and how did they contribute?"   
# Finding feature importance is different for each machine learning model.  
# One way to find feature importance is to search for "(MODEL NAME) feature importance".  
# Let's find the feature importance for our LogisticRegression model...

# %%
# Fit an instance of LogisticRegression
# gs_log_reg.best_params_
clf = LogisticRegression(C=0.20433597178569418,
                         solver='liblinear')
clf.fit(X_train, y_train);

# %%
# Check coef_
clf.coef_

# %%
# Match coef's of features to columns
feature_dict = dict(zip(df.columns, list(clf.coef_[0])))
feature_dict

# %%
# Visualize feature importance
feature_df = pd.DataFrame(feature_dict, index=[0])
feature_df.T.plot.bar(title='Feature Importance', legend=False);

# %% [markdown]
# ## 6. Experimentation
# If you haven't hit your evaluation metric yet. ask youself ...   
# * Could you collect more data ?
# * Could you try a better model ? Like CatBoost or XGBoost ?
# * Could you improve the current models ? (beyond what we've done so far)
# * If your model is good enough (you have hit your evaluation metric) how would you export it and share it with others ?
#     
