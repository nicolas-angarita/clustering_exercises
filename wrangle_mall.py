import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt

import os
from env import get_connection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import warnings
warnings.filterwarnings('ignore')




def get_mall_data():
    '''
    This function is used to get zillow data from sql database.
    '''

    if os.path.isfile('mall.csv'):
        return pd.read_csv('mall.csv')
    else:
        url = get_connection('mall_customers')
        
        query = '''
                SELECT *
                FROM customers
                '''
     
        df = pd.read_sql(query, url)
        
        df.to_csv('mall.csv', index = False )
        
        return df    
    
    
def dist_df(df):
    
    num_cols = df.select_dtypes(include = ('float64', 'int64'))

    for col in num_cols:
    
        plt.hist(df[col])
        plt.title(f'distribution of {col}')
        plt.xlabel(f'{col}')
        plt.show()
        

        
        
def train_val_test(df, stratify = None):
    seed = 22
    
    ''' This function is a general function to split our data into our train, validate, and test datasets. We put in a dataframe
    and our target variable to then return us the datasets of train, validate and test.'''
    
    train, test = train_test_split(df, train_size = 0.7, random_state = seed, stratify = None)
    
    validate, test = train_test_split(test, test_size = 0.5, random_state = seed, stratify = None)
    
    return train, validate, test        
        

def mvp_scaled_data(train, 
               validate, 
               test, 
               columns_to_scale = ['age', 'annual_income'],
               return_scaler=False):
    '''
    Scales the 3 data splits. 
    Takes in train, validate, and test data splits and returns their scaled counterparts.
    If return_scalar is True, the scaler object will be returned as well
    '''
    # make copies of our original data so we dont gronk up anything
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()
    #     make the thing
    mms = MinMaxScaler()
    #     fit the thing
    mms.fit(train[columns_to_scale])
    # applying the scaler:
    train_scaled[columns_to_scale] = pd.DataFrame(mms.transform(train[columns_to_scale]),
                                                  columns=train[columns_to_scale].columns.values).set_index([train.index.values])
                                                  
    validate_scaled[columns_to_scale] = pd.DataFrame(mms.transform(validate[columns_to_scale]), 
                                                     columns=validate[columns_to_scale].columns.values).set_index([validate.index.values])
    
    test_scaled[columns_to_scale] = pd.DataFrame(mms.transform(test[columns_to_scale]),
                                                 columns=test[columns_to_scale].columns.values).set_index([test.index.values])
    
    if return_scaler:
        return scaler, train_scaled, validate_scaled, test_scaled
    else:
        return train_scaled, validate_scaled, test_scaled
    
