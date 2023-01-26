import pandas as pd
import numpy as np 

import matplotlib.pyplot as plt

import os
from env import get_connection

import warnings
warnings.filterwarnings('ignore')




def get_zillow_data():
    '''
    This function is used to get zillow data from sql database.
    '''

    if os.path.isfile('zillow.csv'):
        return pd.read_csv('zillow.csv')
    else:
        url = get_connection('zillow')
        
        test = '%'
        
        query = (f'''
                SELECT * 
                FROM properties_2017
                LEFT JOIN airconditioningtype USING(airconditioningtypeid)
                LEFT JOIN architecturalstyletype USING(architecturalstyletypeid)
                LEFT JOIN buildingclasstype USING(buildingclasstypeid)
                LEFT JOIN heatingorsystemtype USING(heatingorsystemtypeid)
                LEFT JOIN predictions_2017 USING(parcelid)
                LEFT JOIN propertylandusetype USING(propertylandusetypeid)
                LEFT JOIN storytype USING(storytypeid)
                LEFT JOIN typeconstructiontype USING(typeconstructiontypeid)
                WHERE transactiondate LIKE ‘2017{test}{test}’
                AND latitude IS NOT NULL
                AND longitude IS NOT NULL;
                ''')
        df = pd.read_sql(query, url)
        
        df.to_csv('zillow.csv', index = False )
        
        return df    
    
    
def drop_duplicates(df):
    
    df = df.sort_values('transactiondate')
    
    df = df[df.duplicated(subset=['parcelid'], keep = 'last') == False]
    
    return df


def dist_df(df):
    num_cols = df.select_dtypes(include = ('float64'))

    for col in num_cols:
    
        plt.hist(df[col])
        plt.title(f'distribution of {col}')
        plt.xlabel(f'{col}')
        plt.show()

        
def missing_values(df):
    
    missing_df = pd.DataFrame(df.isna().sum(), columns=['num_rows_missing'])
    missing_df['pct_rows_missing'] = missing_df['num_rows_missing'] / len(df)
    
    return missing_df


def sfh(df):
    
    sp = [261, 266, 263, 275, 264]
    df = df[df['propertylandusetypeid'].isin(sp)]
    
    return df    


def handle_missing_values(df, prop_required_col, prop_required_row):
    
    drop_cols = round(prop_required_col * len(df))
    df.dropna(thresh=drop_cols, axis=1, inplace=True)
    
    drop_rows = round(prop_required_row * len(df.columns))
    df.dropna(thresh=drop_rows, axis=0, inplace=True)
    
    return df    
   

def wrangle_zillow(prop_required_col, prop_required_row):
    
    df = get_zillow_data()
    df = drop_duplicates(df)
    df = sfh(df)
    df = handle_missing_values(df, prop_required_col, prop_required_row)
    
    return df