# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 22:41:16 2023

@author: tashf
"""


#### Part 1(a), Building the Classificaiton model
#### Part 1(b), Test the model on grading dataset
#### Part 2, Clustering


############
### Note ###
############
# Part 1(a) and Part 1(b) both have to be run to get the accuracy as in Part 1(a) model is trained 
#and that best trained model is used in part1(b).

############################
######   Part 1(a)  ########
############################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


##########################
### Data Preprocessing ###
##########################

# Import the data
kickstarter = pd.read_excel(r"D:\Google Drive\McGill\Fall Semester\INSY 662\Individual Project\Dataset\Kickstarter.xlsx")


# Columns to keep for the model

column_list = ['goal',  'state', 'static_usd_rate',
        'category', 'name_len', 'name_len_clean', 'blurb_len',
        'blurb_len_clean', 'deadline_weekday', 'created_at_weekday',
        'launched_at_weekday', 'deadline_month',
        #'deadline_day', #'deadline_hr',
        'created_at_month', #'created_at_day', #'created_at_hr',
        'launched_at_month', #'launched_at_day',
        #'launched_at_hr', 
        'create_to_launch_days', 'launch_to_deadline_days']


# Filter Columns
kickstarter=kickstarter[column_list]

# Select Successful and Failed Projects only
kickstarter = kickstarter[kickstarter.state.isin(['successful','failed'])]
kickstarter.state.value_counts() # Counts of project states

# Convert 'goal' to USD
kickstarter['converted_currency_USD'] = kickstarter['goal'] * kickstarter['static_usd_rate']

# Drop the 'goal' and 'static_usd_rate' columns as conversion has been made to USD
kickstarter = kickstarter.drop(['goal', 'static_usd_rate'], axis=1)

# Visualize Dataset
with pd.option_context('display.max_columns',None):
    print(kickstarter.head())


# Distribution of categorical variables
kickstarter.category.value_counts()
#kickstarter.country.value_counts()

# Check for nulls 
kickstarter.isnull().sum().sort_values(ascending=False) # Category has 1254 null values from a total of 13435 ~ 9.3%

# Possible methods to address nulls
#1 Replace as Missing
def missing(df,var):
    return(df[var].fillna('Missing'))

#2 Replace by Mode
def mode(df,var):
    mode_column = df[var].mode()[0]
    return(df[var].fillna(mode_column))

##########################################
#### Both ways do not seem realistic #####

# Drop null value
df = kickstarter.dropna(how='any')
df.reset_index(drop=True,inplace=True)

# Trends by each category
df.groupby('category').state.value_counts()
#df.groupby('country').state.value_counts()


# Check for Anomalies in Data
with pd.option_context('display.max_columns',None):
    print(df.describe()) # no anomalies

# Outliiers

# Boxplot for 'name_len_clean'
plt.figure(figsize=(10, 6))
sns.boxplot(x=kickstarter['name_len_clean'])
plt.title('Boxplot of Name Length Clean')
plt.show()

# Boxplot for 'blurb_len_clean'
plt.figure(figsize=(10, 6))
sns.boxplot(x=kickstarter['blurb_len_clean'])
plt.title('Boxplot of Blurb Length Clean')
plt.show()

# Boxplot for 'converted_currency_USD'
plt.figure(figsize=(10, 6))
sns.boxplot(x=kickstarter['converted_currency_USD'])
plt.title('Boxplot of converted_currency_USD')
plt.show()

from scipy import stats
import numpy as np

# Calculate Z-scores for 'name_len_clean' and 'blurb_len_clean'
kickstarter['name_len_clean_z'] = np.abs(stats.zscore(kickstarter['name_len_clean']))
kickstarter['blurb_len_clean_z'] = np.abs(stats.zscore(kickstarter['blurb_len_clean']))
kickstarter['converted_currency_USD_z'] = np.abs(stats.zscore(kickstarter['converted_currency_USD']))

# threshold for identifying outliers
threshold = 3

# Identify outliers in 'name_len_clean'
outliers_name_len = kickstarter[kickstarter['name_len_clean_z'] > threshold]
print("Number of outliers in 'name_len_clean':", len(outliers_name_len))

# Identify outliers in 'blurb_len_clean'
outliers_blurb_len = kickstarter[kickstarter['blurb_len_clean_z'] > threshold]
print("Number of outliers in 'blurb_len_clean':", len(outliers_blurb_len))

# Identify outliers in 'converted_currency_USD'
outliers_USD_len = kickstarter[kickstarter['converted_currency_USD_z'] > threshold]
print("Number of outliers in 'converted_currency_USD':", len(outliers_USD_len))

# Remove rows where 'name_len_clean_z' or 'blurb_len_clean_z' is greater than the threshold
kickstarter = kickstarter[(kickstarter['name_len_clean_z'] <= threshold) & (kickstarter['blurb_len_clean_z'] <= threshold) & (kickstarter['converted_currency_USD_z'] <= threshold)]
kickstarter = kickstarter.drop(['name_len_clean_z', 'blurb_len_clean_z','converted_currency_USD_z'], axis=1)



# Checking for multi-collinearity between numeric variables
numeric_cols = ['converted_currency_USD', 'name_len', 'name_len_clean', 'blurb_len', 'blurb_len_clean', 'create_to_launch_days', 'launch_to_deadline_days']
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, vmin=corr_matrix.values.min(), vmax=1, square=True, cmap="YlGnBu", linewidths=0.1, annot=True, annot_kws={"fontsize":9})  
plt.show()

# Remove correlated variables: name_len, name_len_clean with 94% & blurb_len, blurb_len_clean with 78%
df.drop(['name_len', 'blurb_len'], axis=1, inplace=True)

# Dummifying categorical variables
dummy_columns = ['category', 'deadline_weekday', 'created_at_weekday', 'launched_at_weekday', 'deadline_month' 
                 #'deadline_day'
                 , 'created_at_month', 
                 #'created_at_day', #'created_at_hr',
                 'launched_at_month' 
                 #'launched_at_day'
                 ]
#, 'launched_at_hr' ,'deadline_hr'
df = pd.get_dummies(df, columns= dummy_columns,drop_first=True)

# Target Variable and Predictors
df['state']=df['state'].map({'successful':'1', 'failed': '0'})
X=df.drop('state',axis=1)
y = df['state']
y = y.astype(int)


# Scaleing the data
from sklearn.preprocessing import StandardScaler
transformed_cols = ['converted_currency_USD', 'name_len_clean', 'blurb_len_clean', 'create_to_launch_days', 'launch_to_deadline_days']
scaler = StandardScaler().fit(X[transformed_cols].values)
feature = scaler.transform(X[transformed_cols].values)
X_std1 = X.copy()
X_std1[transformed_cols] = feature

# Convert to array
X_std=X_std1.values


# Feature Importance
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=6)
model=rf.fit(X_std,y)
feature_imp = pd.DataFrame(list(zip(X.columns,model.feature_importances_)), columns =['predictor','feature importance']).sort_values('feature importance', ascending=False)
feature_imp['feature importance'] = feature_imp['feature importance']*100
#remove country, 'launched_at_hr' ,'deadline_hr,'created_at_hr' as have low feature importance

##########################
## Find Best Parameters ##
##########################

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Logistic Regression
from sklearn.linear_model import LogisticRegression
# logreg = LogisticRegression(max_iter = 15000, random_state = 6)
# score = cross_val_score(logreg, X_std, y, scoring='f1_macro',cv=5)
# print('Logistic Regression: ', np.mean(score), "\n", score)

#KNN
from sklearn.neighbors import KNeighborsClassifier
# knn = KNeighborsClassifier(p = 2)
# parameters_knn = {'n_neighbors': np.arange(2, 40)} 
# GS_knn = GridSearchCV(knn, param_grid = parameters_knn, n_jobs=-1, verbose = True, scoring='f1_macro',cv=5) 
# GS_knn.fit(X_std, y)
# print(f"KNN best parameters are:{GS_knn.best_params_}, Score:{GS_knn.best_score_}")
# knn_best = GS_knn.best_params_
# knn_best_score = GS_knn.best_score_

# Random Forest
from sklearn.ensemble import RandomForestClassifier
# rf = RandomForestClassifier(random_state = 6, warm_start=True)
# parameters_rf = {'n_estimators': np.array([1000, 2000, 3000])} 
# GS_rf = GridSearchCV(rf, param_grid = parameters_rf, n_jobs=-1, verbose = True, scoring='f1_macro',cv=5) 
# GS_rf.fit(X_std, y)
# print(f"Random Forest best parameters:{GS_rf.best_params_}, Score:{GS_rf.best_score_}")
# rfc_best = GS_rf.best_params_
# rfc_best_score = GS_rf.best_score_

# Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
# gbt = GradientBoostingClassifier(random_state=6, warm_start= True)  
# parameters_gbt = {'n_estimators': np.array([1000, 1500, 2000])} 
# gbt_rf = GridSearchCV(gbt, param_grid = parameters_gbt, n_jobs=-1, verbose = True, scoring='f1_macro',cv=5) 
# gbt_rf.fit(X_std, y)
# print(f"Gradient Boosting best parameters:{gbt_rf.best_params_}, Score:{gbt_rf.best_score_}")
# gbt_best = gbt_rf.best_params_
# gbt_best_score = gbt_rf.best_score_


# ANN
from sklearn.neural_network import MLPClassifier
# ann = MLPClassifier(max_iter=1000)
# parameters_ann = {
#     'hidden_layer_sizes': [
#         (50),       # 1 layer with 50 nodes
#         (100),      # 1 layer with 100 nodes
#         (100,100)
#     ],
#     'activation': ['tanh', 'relu'],
#     'alpha': [0.0001, 0.05],
#     'learning_rate': ['constant', 'adaptive']
# } 

# GS_ann = GridSearchCV(ann, param_grid=parameters_ann, n_jobs=-1, verbose=True, scoring='f1_macro', cv=5)
# GS_ann.fit(X_std, y)
# print(f"ANN best parameters: {GS_ann.best_params_}, Score: {GS_ann.best_score_}")

# ann_best = GS_ann.best_params_
# ann_best_score = GS_ann.best_score_



######################################################################
# Comparing results for  classification models using best parameters #
######################################################################

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, cross_validate

# log_model= LogisticRegression(max_iter = 15000, random_state = 6)
# rf_model = RandomForestClassifier(random_state=6, warm_start=False, n_estimators=2000)
# ann_model = MLPClassifier(max_iter=1000, hidden_layer_sizes = (100), activation = 'tanh', alpha=0.05, learning_rate = 'constant')
# knn_model = KNeighborsClassifier(n_neighbors = 3, p = 2)
# gbt_model = GradientBoostingClassifier(random_state=6, warm_start= True,n_estimators=1000)  

# models = [log_model,
#           rf_model,
#           ann_model,
#           knn_model,
#           gbt_model
          
# ]

# names = ["Logistic Regression", "Random Forest", "Artificial Neural Network","KNN Model", "Gradient Boosting"]

# results = []
# for model, name in zip(models, names):
#     print(name)
#     scores = cross_validate(model, X=X_std, y=y, cv=3, scoring=('accuracy', 'precision', 'recall'), return_train_score=False)
#     print(scores)
#     result = [np.average(scores[x]) for x in scores.keys()]
#     results.append(result)

# k = scores.keys()
# models_comparison = pd.DataFrame(results, index = [name for name in names], columns = [x for x in k])
# models_comparison.to_csv('models_comparison.csv')

## Train the model
from sklearn.ensemble import GradientBoostingClassifier
gbt_model = GradientBoostingClassifier(random_state=6, warm_start= True,n_estimators=1000)  
gbt_model.fit(X_std, y)

#############################################################################################################################
###### Models were recursively trained for  categorical predictors  by removing one by one to find their feature importance #######
#############################################################################################################################

#from sklearn.model_selection import train_test_split
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import recall_score

# Split the data into training and testing sets (70% train, 30% test)
#X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size=0.3, random_state=6)

# Create and fit the RandomForest model
#gbt_model = GradientBoostingClassifier(random_state=6, warm_start=False, n_estimators=1000)
#gbt_model.fit(X_train, y_train)

# Make predictions on the test set
#y_pred = gbt_model.predict(X_test)

# Calculate and print the accuracy
##accuracy = accuracy_score(y_test, y_pred)
#recall = recall_score(y_test, y_pred)
#print(f"Model Accuracy: {accuracy}")
#print(f"Model Recall: {recall}")


##################
##################
###  Part1(b)  ###
##################
##################
import pandas as pd
import numpy as np
from scipy import stats


########################
## Data Preprocessing ##
########################
# Import Grading Dataset (Change path here)
kickstarter_grading= pd.read_excel(r"D:\Google Drive\McGill\Fall Semester\INSY 662\Individual Project\Dataset\Kickstarter-Grading-Sample.xlsx")


# Columns to keep 
column_list = ['goal',  'state', 'static_usd_rate',
        'category', 'name_len', 'name_len_clean', 'blurb_len',
        'blurb_len_clean', 'deadline_weekday', 'created_at_weekday',
        'launched_at_weekday', 'deadline_month',
        #'deadline_day', #'deadline_hr',
        'created_at_month', #'created_at_day', #'created_at_hr',
        'launched_at_month',# 'launched_at_day',
        #'launched_at_hr', 
        'create_to_launch_days', 'launch_to_deadline_days']


# Filter Columns
kickstarter_grading=kickstarter_grading[column_list]

# Select Successful and Failed Projects only
kickstarter_grading = kickstarter_grading[kickstarter_grading.state.isin(['successful','failed'])]

# Convert 'goal' to USD
kickstarter_grading['converted_currency_USD'] = kickstarter_grading['goal'] * kickstarter_grading['static_usd_rate']

# Drop the 'goal' and 'static_usd_rate' columns as conversion has been made to USD
kickstarter_grading = kickstarter_grading.drop(['goal', 'static_usd_rate'], axis=1)


# Calculate Z-scores for 'name_len_clean' and 'blurb_len_clean'
kickstarter_grading['name_len_clean_z'] = np.abs(stats.zscore(kickstarter_grading['name_len_clean']))
kickstarter_grading['blurb_len_clean_z'] = np.abs(stats.zscore(kickstarter_grading['blurb_len_clean']))
kickstarter_grading['converted_currency_USD_z'] = np.abs(stats.zscore(kickstarter_grading['converted_currency_USD']))

# threshold for identifying outliers
threshold = 3

# Identify outliers in 'name_len_clean'
outliers_name_len = kickstarter_grading[kickstarter_grading['name_len_clean_z'] > threshold]
print("Number of outliers in 'name_len_clean':", len(outliers_name_len))

# Identify outliers in 'blurb_len_clean'
outliers_blurb_len = kickstarter_grading[kickstarter_grading['blurb_len_clean_z'] > threshold]
print("Number of outliers in 'blurb_len_clean':", len(outliers_blurb_len))

# Identify outliers in 'converted_currency_USD'
outliers_USD_len = kickstarter_grading[kickstarter_grading['converted_currency_USD_z'] > threshold]
print("Number of outliers in 'converted_currency_USD':", len(outliers_USD_len))

# Remove rows where 'name_len_clean_z' or 'blurb_len_clean_z' is greater than the threshold
kickstarter_grading = kickstarter_grading[(kickstarter_grading['name_len_clean_z'] <= threshold) & (kickstarter_grading['blurb_len_clean_z'] <= threshold) & (kickstarter_grading['converted_currency_USD_z'] <= threshold)]
kickstarter_grading = kickstarter_grading.drop(['name_len_clean_z', 'blurb_len_clean_z','converted_currency_USD_z'], axis=1)

# Drop null value
df_grading = kickstarter_grading.dropna(how='any')
df_grading.reset_index(drop=True,inplace=True)

# Remove correlated variables: name_len, name_len_clean with 94% & blurb_len, blurb_len_clean with 78%
df_grading.drop(['name_len', 'blurb_len'], axis=1, inplace=True)

# Dummifying categorical variables
dummy_columns_grading = ['category', 'deadline_weekday', 'created_at_weekday', 'launched_at_weekday', 'deadline_month',# 'deadline_day', 
                         'created_at_month',# 'created_at_day', #'created_at_hr',
                 'launched_at_month' #, 'launched_at_day'
                 ]
df_grading = pd.get_dummies(df_grading, columns= dummy_columns_grading,drop_first=True)

# Target Variable and Predictors
df_grading['state']=df_grading['state'].map({'successful':'1', 'failed': '0'})
X_grading=df_grading.drop('state',axis=1)
y_grading = df_grading['state']
y_grading = y_grading.astype(int)

# Scaling the data
from sklearn.preprocessing import StandardScaler
transformed_cols = ['converted_currency_USD', 'name_len_clean', 'blurb_len_clean', 'create_to_launch_days', 'launch_to_deadline_days']
scaler = StandardScaler().fit(X_grading[transformed_cols].values)
feature = scaler.transform(X_grading[transformed_cols].values)
X_std_grading = X_grading.copy()
X_std_grading[transformed_cols] = feature



columns_grading = set(X_std_grading.columns)
columns_train = set(X_std1.columns)


# Add missing dummy columns to grading dataset
for column in columns_train:
    if column not in columns_grading:
        X_std_grading[column] = 0

# Reorder the columns in grading dataset to match the training dataset
X_std_grading = X_std_grading[X_std1.columns]

# Conver to Array
X_std_grading = X_std_grading.values

##################
## Make predictions using the best model
y_pred_grading = gbt_model.predict(X_std_grading)


from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
accuracy = accuracy_score(y_grading, y_pred_grading)
recall = recall_score(y_grading, y_pred_grading)
print(f"Model Accuracy is {accuracy}")
print(f"Model Recall: {recall}")



############
## Part 2 ##
############
# Import the dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

kickstarter_clustering = pd.read_excel(r"D:\Google Drive\McGill\Fall Semester\INSY 662\Individual Project\Dataset\Kickstarter.xlsx")

kickstarter_clustering.dtypes

# Columns interesting in clustering

filter_list = ['goal','state','usd_pledged','country',
               'staff_pick','static_usd_rate','category','spotlight','name_len_clean'
               ,'blurb_len_clean','create_to_launch_days','launch_to_deadline_days',
               'deadline_yr','launched_at_yr']

kickstarter_clustering=kickstarter_clustering[filter_list]
# Convert goal currency into USD and drop goal and static_usd_rate column

kickstarter_clustering['converted_currency_USD'] = kickstarter_clustering['goal']*kickstarter_clustering['static_usd_rate']

kickstarter_clustering=kickstarter_clustering.drop(['goal','static_usd_rate'],axis=1)

# Drop Null Values
kickstarter_clustering=kickstarter_clustering.dropna()

# Keep only succussfull and failed states 
kickstarter_clustering=kickstarter_clustering[kickstarter_clustering['state'].isin(['successful','failed'])]

# Visualize Continuous Variables and Remove Outliers using Z-Statistic
# Continuous Variables: 
kickstarter_clustering.dtypes

#Covert deadline_yr and launched_at_yr to object
kickstarter_clustering['deadline_yr'] = kickstarter_clustering['deadline_yr'].astype('object')
kickstarter_clustering['launched_at_yr'] = kickstarter_clustering['launched_at_yr'].astype('object')

continuous_variable =kickstarter_clustering.select_dtypes(include=['int64','float64']).columns
continuous_variable
len(continuous_variable) #8 columns numerical so 7 columns categorical

# Plot Box plot
for col in continuous_variable:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=kickstarter_clustering[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Z- Statistics
kickstarter_clustering['name_len_clean_z'] = np.abs(stats.zscore(kickstarter_clustering['name_len_clean']))
kickstarter_clustering['blurb_len_clean_z'] = np.abs(stats.zscore(kickstarter_clustering['blurb_len_clean']))
kickstarter_clustering['converted_currency_USD_z'] = np.abs(stats.zscore(kickstarter_clustering['converted_currency_USD']))
kickstarter_clustering['usd_pledged_z'] = np.abs(stats.zscore(kickstarter_clustering['usd_pledged']))
kickstarter_clustering['create_to_launch_days_z'] = np.abs(stats.zscore(kickstarter_clustering['create_to_launch_days']))
kickstarter_clustering['launch_to_deadline_days_z'] = np.abs(stats.zscore(kickstarter_clustering['launch_to_deadline_days']))

threshold = 3 # Define Threshold

# Identify outliers in 'converted_currency_USD'
outliers_USD_len = kickstarter_clustering[kickstarter_clustering['converted_currency_USD_z'] > threshold]
print("Number of outliers in 'converted_currency_USD':", len(outliers_USD_len))

outliers_name_len_clean_z = kickstarter_clustering[kickstarter_clustering['name_len_clean_z'] > threshold]
print("Number of outliers in 'name_len_clean':", len(outliers_name_len_clean_z))

outliers_blurb_len_clean_z = kickstarter_clustering[kickstarter_clustering['blurb_len_clean_z'] > threshold]
print("Number of outliers in 'blurb_len_clean':", len(outliers_blurb_len_clean_z))

outliers_usd_pledged_z = kickstarter_clustering[kickstarter_clustering['usd_pledged_z'] > threshold]
print("Number of outliers in 'usd_pledged':", len(outliers_usd_pledged_z))

outliers_create_to_launch_days_z = kickstarter_clustering[kickstarter_clustering['create_to_launch_days_z'] > threshold]
print("Number of outliers in 'create_to_launch_days':", len(outliers_create_to_launch_days_z))

outliers_launch_to_deadline_days_z = kickstarter_clustering[kickstarter_clustering['launch_to_deadline_days_z'] > threshold]
print("Number of outliers in 'launch_to_deadline_days':", len(outliers_launch_to_deadline_days_z))


# Remove rows where 'name_len_clean_z' or 'blurb_len_clean_z' is greater than the threshold
kickstarter_clustering = kickstarter_clustering[(kickstarter_clustering['name_len_clean_z'] <= threshold)
                          & (kickstarter_clustering['blurb_len_clean_z'] <= threshold)
                          & (kickstarter_clustering['converted_currency_USD_z'] <= threshold)
                          & (kickstarter_clustering['usd_pledged_z'] <= threshold)
                          & (kickstarter_clustering['create_to_launch_days_z'] <= threshold)
                          & (kickstarter_clustering['launch_to_deadline_days_z'] <= threshold)
                          ]
# Drop the Z Score Columns
kickstarter_clustering = kickstarter_clustering.drop(['name_len_clean_z', 
                                                      'blurb_len_clean_z',
                                                      'converted_currency_USD_z'
                                                      ,'usd_pledged_z'
                                                      ,'create_to_launch_days_z'
                                                      ,'launch_to_deadline_days_z'], axis=1)


# Dummify
kickstarter_clustering['staff_pick'] = kickstarter_clustering['staff_pick'].astype('object')
kickstarter_clustering['spotlight'] = kickstarter_clustering['spotlight'].astype('object')

X_dummy = pd.get_dummies(kickstarter_clustering,columns=['state','country','staff_pick','category','spotlight','deadline_yr','launched_at_yr'],drop_first=False)

# Standardize using MinMax Scaler
from sklearn.preprocessing import MinMaxScaler
transformed_cols = ['converted_currency_USD', 'name_len_clean', 'blurb_len_clean', 'create_to_launch_days', 'launch_to_deadline_days','usd_pledged']
scaler = MinMaxScaler().fit(X_dummy[transformed_cols].values)
feature = scaler.transform(X_dummy[transformed_cols].values)
X_std = X_dummy.copy()
X_std[transformed_cols] = feature


# # Standardize using MinMax Scaler for K Prototype
# from sklearn.preprocessing import MinMaxScaler
# transformed_cols = ['converted_currency_USD', 'name_len_clean', 'blurb_len_clean', 'create_to_launch_days', 'launch_to_deadline_days','usd_pledged']
# scaler = MinMaxScaler().fit(kickstarter_clustering[transformed_cols].values)
# feature = scaler.transform(kickstarter_clustering[transformed_cols].values)
# X_std_K_prot= kickstarter_clustering.copy()
# X_std_K_prot[transformed_cols]=feature


#categorical_columns_list=['state','country','staff_pick','category','spotlight','deadline_yr','launched_at_yr']
#categorical_columns=kickstarter_clustering[categorical_columns_list]


#Silhoutte Method Kprototype
# from kmodes.kprototypes import KPrototypes
# from sklearn.metrics import silhouette_score
# for i in range (3,11):    
#     kmixed = KPrototypes(n_clusters=i)
#     model = kmixed.fit(X_std_K_prot,categorical=[0,2,3,4,5,10,11])
#     labels = model.labels_
#     print(i,':',silhouette_score(X_std_K_prot,labels))

### Silhoute does not work for K prototype because this method requires numerical features only to calculate distance between observations

#Elbow Mthod KPrototype
# from kmodes.kprototypes import KPrototypes
# import matplotlib.pyplot as plt

# costs = []
# range_n_clusters = range(2, 11)

# for n_clusters in range_n_clusters:
#     kproto = KPrototypes(n_clusters=n_clusters, init='Cao')
#     model = kproto.fit(X_std_K_prot, categorical=[0,2,3,4,5,10,11])
#     costs.append(model.cost_)

# plt.plot(range_n_clusters, costs, marker='o')
# plt.title('Elbow Method For Optimal Number of Clusters')
# plt.xlabel('Number of Clusters')
# plt.ylabel('Total Cost')
# plt.show()

# Optimal K found is 5
# kmixed = KPrototypes(n_clusters=5)
# cluster = kmixed.fit_predict(X_std_K_prot, categorical=[0,2,3,4,5,10,11]) #specify the categorical columns

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#     print (pd.DataFrame(kmixed.cluster_centroids_, columns=X_std_K_prot.columns))
    
# Interpratabilty of K-Means is more intuitive and straightforwad so will use that

############# K-Means #############
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# Elbow method to determine K
withinss = []
for i in range (3,11):    
    kmeans = KMeans(n_clusters=i)
    model = kmeans.fit(X_std)
    withinss.append(model.inertia_)

plt.plot([3,4,5,6,7,8,9,10],withinss) # probably k = 5

# Silhouette method to determine K
from sklearn.metrics import silhouette_score
for i in range (3,11):    
    kmeans = KMeans(n_clusters=i)
    model = kmeans.fit(X_std)
    labels = model.labels_
    print(i,':',silhouette_score(X_std,labels)) # k = 5 

# Pseudo F score to determine K
from sklearn.metrics import calinski_harabasz_score
for i in range (3,11):    
    kmeans = KMeans(n_clusters=i)
    model = kmeans.fit(X_std)
    labels = model.labels_
    score = calinski_harabasz_score(X_std, labels)
    print(i,'F-score:',score) # k = 3


# # Silhouette method to determine best parameters for DBCAN
# import numpy as np
# from sklearn.metrics import silhouette_score
# from sklearn.cluster import DBSCAN

# for i in range(3, 11):
#     for j in np.arange(0.1, 1, 0.1):
#         dbscans = DBSCAN(eps=j, min_samples=i)
#         model = dbscans.fit(X_std)
#         labels = model.labels_
        
#         # Check if all labels are the same (silhouette_score is not defined for a single cluster)
#         if len(set(labels)) == 1:
#             print(i, j, ':', 'Single cluster found, silhouette score is not defined')
#         else:
#             score = silhouette_score(X_std, labels)
#             print(i, j, ':', score)

# #####################
# # Number of Cluster #
# dbscan = DBSCAN(eps=0.9, min_samples=3)
# dbscan.fit(X_std)

# # Extract labels
# labels = dbscan.labels_

# # Count the number of clusters
# n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

# print(f"Number of clusters formed: {n_clusters}")
# print(f"Number of noise points: {list(labels).count(-1)}")

## 509 clusters being formed interpretation is difficult so use K-means

# Final K-means
from sklearn.cluster import KMeans
model = KMeans(n_clusters=5)
cluster = model.fit_predict(X_std)


with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print (pd.DataFrame(model.cluster_centers_, columns=X_std.columns))
X_dummy['Cluster'] = cluster


############# Examining Clusters: Example #############
import seaborn as sns

# Distribution of the clusters
pl = sns.countplot(x=X_dummy["Cluster"])
pl.set_title("Distribution Of The Clusters")
plt.show()

# Income by clusters
pl = sns.boxplot(x="Cluster", y="converted_currency_USD", data=X_dummy)
pl.set_title("Goal by Clusters")
plt.show()


# Assuming 'model' is your trained KMeans model and 'X_std' is your standardized data used for KMeans
results_cluster = pd.DataFrame(data=model.cluster_centers_, columns=X_std.columns)

# Melt the cluster centers DataFrame for visualization
melted_cluster_centers = pd.melt(results_cluster.reset_index(), id_vars=['index'], value_name='Value', var_name='Variable')

# Filter out the continuous variables for plotting
melted_cluster_centers_cont = melted_cluster_centers[melted_cluster_centers['Variable'].isin(continuous_variable)]

# Plotting the continuous variables using a strip plot
sns.set(font_scale=1)
plt.figure(figsize=(10, 6))
sns.stripplot(x='Variable', y='Value', hue='index', data=melted_cluster_centers_cont, 
              jitter=True, dodge=False, palette='bright', size=5)
plt.xticks(rotation=90)
plt.title('Cluster Centers for Continuous Variables')
plt.legend(title='Cluster')
plt.show()

# Compiling information for each cluster
categories = [col for col in X_std.columns if col.startswith("category_")]

cluster_info = {}
for cluster_id in range(5):
    cluster_info[f'Cluster {cluster_id}'] = {cat: f'{results_cluster.at[cluster_id, cat]*100:.2f}%' for cat in categories}

# Displaying the compiled information
for cluster, info in cluster_info.items():
    print(f"{cluster}:")
    for category, value in info.items():
        print(f"  {category}: {value}")











