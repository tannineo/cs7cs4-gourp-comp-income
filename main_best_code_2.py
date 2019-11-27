import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
import lightgbm as lgb

null_values = {
    'Year of Record': ["#N/A"],
    'Housing Situation': ["0", "nA"],
    'Work Experience in Current Job [years]': ["#NUM!"],
    'Satisfation with employer': ["#N/A"],
    'Gender': ["#N/A", "0", "unknown"],
    'Country': ["0"],
    'Profession': ["#N/A"],
    'University Degree': ["0", "#N/A"],
    'Hair Color': ["#N/A", "0", "Unknown"]
}

# load the training dataset
tr_dataset = pd.read_csv('training data.csv', na_values=null_values, low_memory=False)

# drop irrelevant columns
tr_dataset = tr_dataset.drop(['Instance', 'Wears Glasses', 'Hair Color'], axis = 1)

rename_columnsTo = {
    "Yearly Income in addition to Salary (e.g. Rental Income)": "AddnIncome"
}
tr_dataset = tr_dataset.rename(columns=rename_columnsTo)

tr_dataset['AddnIncome'] = tr_dataset.AddnIncome.str.split(' ').str[0].str.strip()
tr_dataset['AddnIncome'] = tr_dataset['AddnIncome'].astype('float64')

# removing duplicate entries, if present
if(len(tr_dataset[tr_dataset.duplicated()].index > 0)):
    tr_dataset = tr_dataset.drop_duplicates()

# finding the missing data from the training dataset using the heatmap
sb.heatmap(tr_dataset.isnull(), yticklabels=False)
plt.show()

# creating the noise function to avoid overfitting
def addNoise(dataframe, noise_level):
    return dataframe * (1 + noise_level * np.random.rand(len(dataframe)))

# encoding the dataset using target encoding
def targetEncoding(training_set, target_variable, cat_cols, min_sample_leaf, alpha, noise_level):
    tr_target = training_set.copy()
    globalmean = training_set[target_variable].mean()
    cat_mapping = dict()
    default_mapping = dict()

    for column in cat_cols:
        cat_count = training_set.groupby(column).size()
        target_cat_mean = training_set.groupby(column)[target_variable].mean()
        reg_smooth_val = ((target_cat_mean * cat_count) + (globalmean * alpha))/(cat_count + alpha)

        tr_target.loc[:, column] = tr_target[column].map(reg_smooth_val)
        tr_target[column].fillna(globalmean, inplace =True)
        #tr_target[column] = addNoise(tr_target[column], noise_level)

        cat_mapping[column] = reg_smooth_val
        default_mapping[column] = globalmean
    return tr_target, cat_mapping, default_mapping

categorical_columns = ['Housing Situation', 
                       'Satisfation with employer', 
                       'Gender', 
                       'Country', 
                       'Profession', 
                       'University Degree']
tr_targetX, target_mapping, default_mapping = targetEncoding(tr_dataset, 
                                                             'Total Yearly Income [EUR]', 
                                                             categorical_columns,
                                                             100, 
                                                             10, 
                                                             0.05)

# finding the missing data from the encoded dataset using the heatmap
sb.heatmap(tr_targetX.isnull(), yticklabels=False)
plt.show()

# filling the missing values for numerical columns
def fillNA(dataframe, num_cols):
    data = {}
    for column in num_cols:
        data[column] = dataframe[column].mean()
    return dataframe.fillna(value = data)

numerical_columns = ['Year of Record', 
                     'Age', 
                     'Work Experience in Current Job [years]']
tr_targetX = fillNA(tr_targetX, numerical_columns)

tr_X = tr_targetX.iloc[:, :-1]
tr_y = tr_targetX.iloc[:, -1]

# splitting the data into training set & testing set
X_train, X_test, y_train, y_test = train_test_split(tr_X, tr_y, test_size = 0.2, random_state = 1201)

parameters = {
          'max_depth': 30,
          'learning_rate': 0.001,
          "boosting": "gbdt",
          "objective": "tweedie",
          "bagging_seed": 11,
          "metric": 'mae',
          "verbosity": -1,
          "num_threads":4
         }

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test)

regressor = lgb.train(parameters,
                train_data,
                100000,
                valid_sets = [train_data, test_data],
                verbose_eval=1000,
                early_stopping_rounds=500)
y_pred = regressor.predict(X_test)

from sklearn import metrics
print('Mean Absolute Error: ', metrics.mean_absolute_error(y_test, y_pred))

#load the test dataset
ts_dataset = pd.read_csv('test data.csv', na_values=null_values, low_memory=False)
ts_dataset = ts_dataset.drop(['Instance', 'Wears Glasses', 'Hair Color'], axis = 1)

ts_dataset = ts_dataset.rename(columns=rename_columnsTo)

ts_dataset['AddnIncome'] = ts_dataset.AddnIncome.str.split(' ').str[0].str.strip()
ts_dataset['AddnIncome'] = ts_dataset['AddnIncome'].astype('float64')

#finding the missing data from the training dataset using the heatmap
sb.heatmap(ts_dataset.isnull(), yticklabels=False)
plt.show()

#mapping the test dataset with target encoding values
ts_targetX = ts_dataset.copy()
for column in categorical_columns:
    ts_targetX.loc[:, column] = ts_targetX[column].map(target_mapping[column])
    ts_targetX[column].fillna(default_mapping[column], inplace =True)

#filling the missing numerical values in the test dataset
ts_targetX = fillNA(ts_targetX, numerical_columns)

ts_X = ts_targetX.iloc[:, :-1]
ts_y = ts_targetX.iloc[:, -1]

#predicting the dependant variable for the test dataset
ts_y_pred = regressor.predict(ts_X)