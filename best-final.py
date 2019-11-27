#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from math import sqrt, log, exp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from feature_engine import missing_data_imputers as mdi
from sklearn.model_selection import train_test_split
from feature_engine.categorical_encoders import OneHotCategoricalEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pickle 
import feature_engine.missing_data_imputers as mdi





'done'


# In[ ]:





# In[6]:


#salary_data = pd.read_csv("source_data/train.csv", index_col=[0], dtype={
#    'Housing Situation': 'unicode',
#    'Work Experience in Current Job [years]': 'unicode'
#})
#test_data = pd.read_csv("source_data/test.csv", index_col=[0], dtype={
#    'Housing Situation': 'unicode',
#    'Work Experience in Current Job [years]': 'unicode'
#})


#Read in Test Files
salary_data = pd.read_csv("C:/Users/Mary/Desktop/Marys College/Fourth year/Machine learning/Assignment 2/tcd-ml-1920-group-income-train.csv", index_col=[0], dtype={
    'Housing Situation': 'unicode',
    'Work Experience in Current Job [years]': 'unicode'
})
test_data = pd.read_csv(r"C:\Users\Mary\Desktop\Marys College\Fourth year\Machine learning\Assignment 2\tcd-ml-1920-group-income-test.csv", index_col=[0], dtype={
    'Housing Situation': 'unicode',
    'Work Experience in Current Job [years]': 'unicode'
})




# In[108]:


def training_data_preprocessing(dataset):
    #dataset = dataset[dataset['Income in EUR'] < 2500000]
    #logging income value
    dataset['Total Yearly Income [EUR]'] = dataset['Total Yearly Income [EUR]']
    return dataset


# In[8]:


# rid of duplicates
salary_data = salary_data.drop_duplicates()
#logging income value
#salary_data['Total Yearly Income [EUR]'] = salary_data['Total Yearly Income [EUR]'].apply(np.log)
# np.around Income
salary_data['Total Yearly Income [EUR]'] = salary_data['Total Yearly Income [EUR]'].map(lambda i: np.around(i))
#salary_data['Yearly Income in addition to Salary (e.g. Rental Income)'] = salary_data['Yearly Income in addition to Salary (e.g. Rental Income)'].map(lambda i: np.around(i))

# drop (t_df['Year of Record'] < 2000) & (t_df['Total Yearly Income [EUR]'] > 400000)
#salary_data.drop([(t_df['Year of Record'] < 2000) & (salary_data['Total Yearly Income [EUR]'] > 400000)].index)

salary_data.head()
'done'


# In[110]:


salary_data = training_data_preprocessing(salary_data)


# In[13]:


rename_cols = {
    "Year of Record": "Year",
    "Housing Situation": "Housing",
    "Crime Level in the City of Employement": "Crime",
    "Work Experience in Current Job [years]": "Work",
    "Satisfation with employer": "Satisfaction",
    "Size of City": "Size",
    "University Degree": "University",
    "Wears Glasses": "Wears",
    "Hair Color": "Hair",
    "Body Height [cm]": "Body",
    "Yearly Income in addition to Salary (e.g. Rental Income)": "Yearly",
    "Total Yearly Income [EUR]": "Income"
}

salary_data = salary_data.rename(columns=rename_cols)
#graphin
#salary_data.head()


# In[14]:


salary_data.shape


# In[15]:


#train = salary_data[salary_data['Income'] < 400000] # rid of some incomes that are too high
##changed figure value below - mary
train = salary_data[salary_data['Income'] < 400000]
test = test_data.rename(columns=rename_cols) # rename test data


# In[27]:


# set up the imputer-mary
imputer = mdi.RandomSampleImputer(random_state=['Gender', 'Country'],
                          seed='observation',
                          seeding_method='add')
# fit the imputer
imputer.fit(train)

train= imputer.transform(train)
test= imputer.transform(test)
'done'


# In[24]:


#simple imputer to predict misssing values-mary
# set up the imputer
imputer = mdi.FrequentCategoryImputer(variables='Country')
# fit the imputer
imputer.fit(train)

# transform the data
train= imputer.transform(train)
test= imputer.transform(test)
'done'


# In[25]:


train.shape


# In[115]:


data = pd.concat([train,test],ignore_index=True) # combine the test data and train data

data['Yearly'] = data['Yearly'].map(lambda x: float(x.rstrip(' EUR')))


# In[116]:


salary_data['Yearly'] = salary_data['Yearly'].map(lambda x: float(x.rstrip(' EUR')))
salary_data['Housing'] = pd.Categorical(salary_data['Housing']).codes  
salary_data['Work'] = pd.Categorical(salary_data['Work']).codes 
salary_data['Satisfaction'] = pd.Categorical(salary_data['Satisfaction']).codes  
salary_data['Gender'] = pd.Categorical(salary_data['Gender']).codes  
salary_data['Country'] = pd.Categorical(salary_data['Country']).codes 
salary_data['Profession'] = pd.Categorical(salary_data['Profession']).codes
salary_data['University'] = pd.Categorical(salary_data['University']).codes 
salary_data['Hair'] = pd.Categorical(salary_data['Hair']).codes
salary_data['Yearly'] = pd.Categorical(salary_data['Yearly']).codes
##changed figure below - mary
salary_data = salary_data[salary_data['Income'] < 400000]


# In[117]:


#graphin
#salary_data[:120000].plot.scatter(x='Year',y='Income')


# In[118]:


#graphin
#salary_data[:120000].plot.scatter(x='Housing',y='Income')


# In[119]:


#graphin
#salary_data[:120000].plot.scatter(x='Crime',y='Income')


# In[120]:


#graphin
#salary_data[:120000].plot.scatter(x='Work',y='Income')


# In[121]:


#graphin
#salary_data[:120000].plot.scatter(x='Satisfaction',y='Income')


# In[122]:


#grphin
#salary_data[:120000].plot.scatter(x='Gender',y='Income')


# In[123]:


#graphin
#salary_data[:120000].plot.scatter(x='Age',y='Income')


# In[124]:


#graphin
#salary_data[:120000].plot.scatter(x='Country',y='Income')


# In[125]:


#graphin
#salary_data[:120000].plot.scatter(x='Size',y='Income')


# In[126]:


#graphin
#salary_data[:120000].plot.scatter(x='Profession',y='Income')


# In[127]:


#graphin
#salary_data[:120000].plot.scatter(x='University',y='Income')


# In[128]:


#graphin
#salary_data[:120000].plot.scatter(x='Wears',y='Income')


# In[129]:


#graphin
#salary_data[:120000].plot.scatter(x='Hair',y='Income')


# In[130]:


#graphin
#salary_data[:120000].plot.scatter(x='Body',y='Income')


# In[131]:


#graphin
#salary_data[:120000].plot.scatter(x='Yearly',y='Income')


# In[132]:


# devide into 2 columns according to the 1990 boundary
data['Year_B1990'] = data['Year'].map(lambda y: y if y <= 1990 else 0)
data['Year_A1990'] = data['Year'].map(lambda y: y if y > 1990 else 0)

#data.head()


# In[133]:


data.University.mode()


# In[146]:


# fillna with default values
fill_col_dict = {
    #changing year to most common value - from 1977 to 1982 - mary
    'Year': 1982,
    #changing satisfaction from 0.0 to'Average' -mary
    'Satisfaction': 'Average',
    'Gender': 'male',
    #changing country from hondorus to Switzerland -mary
    'Country': 'Switzerland',
   # changing profession from 'payment analyst' -mary
    'Profession': 'postal service mail sorter',
    'University': 'Bachelor',
    'Hair': 'Black',
    }
for col in fill_col_dict.keys():
    data[col] = data[col].fillna(fill_col_dict[col])


# In[147]:


def create_cat_con(df,cats,cons,normalize=True):  
    for i,cat in enumerate(cats):
        vc = df[cat].value_counts(dropna=False, normalize=normalize).to_dict() 
        nm = cat + '_FE_FULL' 
        df[nm] = df[cat].map(vc)
        df[nm] = df[nm].astype('float32')
        for j,con in enumerate(cons):
            new_col = cat +'_'+ con
            print('timeblock frequency encoding:', new_col)
            df[new_col] = df[cat].astype(str)+'_'+df[con].astype(str)  
            temp_df = df[new_col]
            fq_encode = temp_df.value_counts(normalize=True).to_dict()
            df[new_col] = df[new_col].map(fq_encode)
            df[new_col] = df[new_col] / df[cat+'_FE_FULL']
    return df

data.head()


# In[148]:


# cats = ['Year']
# cons = ['Crime']

cats = ['Year_B1990', 'Year_A1990', 'Year', 'Housing', 'Work',
        'Satisfaction', 'Gender', 'Age',
        'Country', 'Profession', 'University', 'Wears', 'Hair']
cons = ['Crime', 'Size', 'Body', 'Yearly']

data = create_cat_con(data,cats,cons)
print('create_cat_con OK!!!')


# In[149]:


data.head()


# In[150]:


for col in train.dtypes[train.dtypes == 'object'].index.tolist():
    feat_le = LabelEncoder()
    feat_le.fit(data[col].unique().astype(str))
    data[col] = feat_le.transform(data[col].astype(str))
    
del_col = set(['Income','Instance'])
features_col =  list(set(data) - del_col)

X_train,X_test = data[features_col].iloc[:880388],data[features_col].iloc[880389:]
Y_train = data['Income'].iloc[:880388]
X_test_id = data.index[880389:]
x_train,x_val,y_train,y_val = train_test_split(X_train,Y_train,test_size=0.2,random_state=1234)


# In[151]:


#this is where lgboost is
params = {
          'max_depth': 20,
          'learning_rate': 0.1,
          "boosting": "gbdt",
          "bagging_seed": 11,
          "metric": 'mse',
          "verbosity": -1,
         }
trn_data = lgb.Dataset(x_train, label=y_train)
val_data = lgb.Dataset(x_val, label=y_val)
# test_data = lgb.Dataset(X_test)
clf = lgb.train(params, trn_data, 100000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds=500)
pre_test_lgb = clf.predict(X_test)
#pre_val_lgb = (pre_val_lgb)
'done'


# In[152]:


from sklearn.metrics import mean_absolute_error
#pre_val_lgb = np.exp(pre_val_lgb)
pre_val_lgb = clf.predict(x_val)
#val_mae = mean_absolute_error(y_val,pre_val_lgb)
val_mae = mean_absolute_error((y_val), (pre_val_lgb))
val_mae


# In[145]:


files = 'tcd-ml-1920-group-income-submission.csv'
df = pd.read_csv(files)

sub_df = pd.DataFrame({'Instance':X_test_id,
                       'Total Yearly Income [EUR]':pre_test_lgb})
#getting rid of minus values - mary
#sub_df[sub_df < 0] = 0
sub_df['Total Yearly Income [EUR]'] = sub_df['Total Yearly Income [EUR]'].abs()
sub_df['Instance'] = sub_df['Instance'].map(lambda i: i - 880388)
sub_df.to_csv("submission2.csv",index=False)

'done'


# In[ ]:




