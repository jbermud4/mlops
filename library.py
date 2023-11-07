"""
Author: Jan Bermudez
Description: A library for pandas data frame transformers
Last Modified: 10/24/2023
"""
import pandas as pd
import numpy as np
import sklearn
sklearn.set_config(transform_output="pandas")  #says pass pandas tables through pipeline instead of numpy matrices
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
import subprocess
import sys
subprocess.call([sys.executable, '-m', 'pip', 'install', 'category_encoders'])  #replaces !pip install
import category_encoders as ce
from sklearn.neighbors import KNeighborsClassifier  #the KNN model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import joblib

titanic_variance_based_split = 107
customer_variance_based_split = 113
########################################################################################################################################################


class CustomMappingTransformer(BaseEstimator, TransformerMixin):

  def __init__(self, mapping_column, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return self

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.mapping_column}"'  #column legit?

    #Set up for producing warnings. First have to rework nan values to allow set operations to work.
    #In particular, the conversion of a column to a Series, e.g., X[self.mapping_column], transforms nan values in strange ways that screw up set differencing.
    #Strategy is to convert empty values to a string then the string back to np.nan
    placeholder = "NaN"
    column_values = X[self.mapping_column].fillna(placeholder).tolist()  #convert all nan values to the string "NaN" in new list
    column_values = [np.nan if v == placeholder else v for v in column_values]  #now convert back to np.nan
    keys_values = self.mapping_dict.keys()

    column_set = set(column_values)  #without the conversion above, the set will fail to have np.nan values where they should be.
    keys_set = set(keys_values)      #this will have np.nan values where they should be so no conversion necessary.

    #now check to see if all keys are contained in the column.
    keys_not_found = keys_set - column_set
    if keys_not_found:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these mapping keys do not appear in the column: {keys_not_found}\n")

    #now check to see if some keys are absent
    keys_absent = column_set -  keys_set
    if keys_absent:
      print(f"\nWarning: {self.__class__.__name__}[{self.mapping_column}] these values in the column do not contain corresponding mapping keys: {keys_absent}\n")

    #do actual mapping
    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    #self.fit(X,y)
    result = self.transform(X)
    return result

########################################################################################################################################################
#This class will rename one or more columns.


class CustomRenamingTransformer(BaseEstimator, TransformerMixin):
  # init method
  def __init__(self, mapping_dict:dict):
    assert isinstance(mapping_dict, dict), f'{self.__class__.__name__} constructor expected dictionary but got {type(mapping_dict)} instead.'
    assert len(mapping_dict) > 0, f'{self.__class__.__name__} constructor requires dictionary to be non-empty.'
    #assert len(set(mapping_dict.values())) == len(mapping_dict), f'{self.__class__.__name__} constructor requires new column names to be unique.'
    self.mapping_dict = mapping_dict

  #define fit to do nothing but give warning
  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return self

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'

    # check if all columns exist and retrieve any bogus column names
    mapping_keys = set(self.mapping_dict.keys())
    x_keys = set(X.columns.to_list())
    keys_not_found = mapping_keys - x_keys

    assert len(keys_not_found) == 0, f'{self.__class__.__name__}.transform unknown column(s) "{keys_not_found}"'  #columns legit?

    #do the actual mapping
    X_ = X.copy()
    X_.rename(columns=self.mapping_dict, inplace=True)
    return X_
    
  def fit_transform(self, X, y = None):
    #self.fit(X,y)
    result = self.transform(X)
    return result

########################################################################################################################################################


class CustomOHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=False):
    self.target_column = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first

  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return self

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'  #column legit?
    #do the actual mapping
    X_ = X.copy()
    X_ = pd.get_dummies(X,
                        prefix=str(self.target_column),
                        prefix_sep='_',
                        columns=[self.target_column], # target column str or int
                        dummy_na=self.dummy_na,       # dummy_na
                        drop_first=self.drop_first    # drop_first
                        )

    return X_

  #write fit_transform that skips fit
  def fit_transform(self, X, y = None):
    #self.fit(X,y)
    result = self.transform(X)
    return result

########################################################################################################################################################


class CustomSigma3Transformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column):
    assert isinstance(target_column, str), f'expected str but got {type(target_column)} instead.'
    self.target_column = target_column
    self.bounds = None

  def fit(self, X, y = None):
    assert isinstance(X, pd.core.frame.DataFrame), f'expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'  #column legit?
    
    mean = X[self.target_column].mean()
    std = X[self.target_column].std()
    lower_bound = mean - (3 * std)
    upper_bound = mean + (3 * std)
    self.bounds = (lower_bound, upper_bound)
    return self

  def transform(self, X):
    assert self.bounds != None, f'NotFittedError: This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'
    minb, maxb = self.bounds
    X_ = X.copy()
    X_[self.target_column] = X[self.target_column].clip(lower=minb, upper=maxb)
    X_.reset_index(drop=True, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    self.fit(X)
    result = self.transform(X)
    return result

########################################################################################################################################################


class CustomTukeyTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, fence='outer'):
    assert fence in ['inner', 'outer']
    assert isinstance(target_column, str), f'{self.__class__.__name__} constructor expected str for target_column but got {type(target_column)} instead.'
    self.fence = fence
    self.target_column = target_column
    self.inner = None
    self.outer = None

  def fit(self, X, y = None):
    assert isinstance(X, pd.core.frame.DataFrame), f'expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'  #column legit?
    q1 = X[self.target_column].quantile(0.25)
    q3 = X[self.target_column].quantile(0.75)
    iqr = q3-q1
    # inner fence
    inner_low = q1 - (1.5*iqr)
    inner_high = q3 + (1.5*iqr)
    self.inner = (inner_low, inner_high)
    # outer fence
    outer_low = q1 - (3*iqr)
    outer_high = q3 + (3*iqr)
    self.outer = (outer_low, outer_high)
    
    return self

  def transform(self, X):
    assert self.inner != None, f'NotFittedError: This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'
    # unpack user specified fence
    minb, maxb = self.inner if self.fence == "inner" else self.outer
    X_ = X.copy()
    X_[self.target_column] = X[self.target_column].clip(lower=minb, upper=maxb)
    X_.reset_index(drop=True, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    self.fit(X)
    result = self.transform(X)
    return result
    
########################################################################################################################################################


class CustomRobustTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column):
    self.target_column = column
    self.median = None
    self.iqr = None

  def fit(self, X, y = None):
    self.median = X[self.target_column].median()
    self.iqr = X[self.target_column].quantile(.75) - X[self.target_column].quantile(.25)
    return self

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.median != None, f'NotFittedError: This {self.__class__.__name__} instance is not fitted yet. Call "fit" with appropriate arguments before using this estimator.'
    assert self.target_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'  #column legit?
    #do the actual mapping
    X_ = X.copy()
    X_[self.target_column] -= self.median
    X_[self.target_column] /= self.iqr
    return X_

  def fit_transform(self, X, y = None):
    self.fit(X,y)
    result = self.transform(X)
    return result

########################################################################################################################################################

def find_random_state(features_df, labels, n=200):
  var = []  #collect test_error/train_error where error based on F1 score
  for i in range(1, n):
    train_X, test_X, train_y, test_y = train_test_split(features_df, labels, test_size=0.2, shuffle=True,
                                                    random_state=i, stratify=labels)
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(train_X, train_y)  #train model
    train_pred = model.predict(train_X)           #predict against training set
    test_pred = model.predict(test_X)             #predict against test set
    train_f1 = f1_score(train_y, train_pred)   #F1 on training predictions
    test_f1 = f1_score(test_y, test_pred)      #F1 on test predictions
    f1_ratio = test_f1/train_f1          #take the ratio
    var.append(f1_ratio)
  rs_value = sum(var)/len(var)  #get average ratio value
  return np.array(abs(var - rs_value)).argmin()

########################################################################################################################################################


titanic_transformer = Pipeline(steps=[
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('map_class', CustomMappingTransformer('Class', {'Crew': 0, 'C3': 1, 'C2': 2, 'C1': 3})),
    ('target_joined', ce.TargetEncoder(cols=['Joined'],
                           handle_missing='return_nan', #will use imputer later to fill in
                           handle_unknown='return_nan'  #will use imputer later to fill in
    )),
    ('tukey_age', CustomTukeyTransformer(target_column='Age', fence='outer')),
    ('tukey_fare', CustomTukeyTransformer(target_column='Fare', fence='outer')),
    ('scale_age', CustomRobustTransformer('Age')),  #from chapter 5
    ('scale_fare', CustomRobustTransformer('Fare')),  #from chapter 5
    ('imputer', KNNImputer(n_neighbors=5, weights="uniform", add_indicator=False))  #from chapter 6
    ], verbose=True)

########################################################################################################################################################


customer_transformer = Pipeline(steps=[
    ('map_os', CustomMappingTransformer('OS', {'Android': 0, 'iOS': 1})),
    ('target_isp', ce.TargetEncoder(cols=['ISP'],
                           handle_missing='return_nan', #will use imputer later to fill in
                           handle_unknown='return_nan'  #will use imputer later to fill in
    )),
    ('map_level', CustomMappingTransformer('Experience Level', {'low': 0, 'medium': 1, 'high':2})),
    ('map_gender', CustomMappingTransformer('Gender', {'Male': 0, 'Female': 1})),
    ('tukey_age', CustomTukeyTransformer('Age', 'inner')),  #from chapter 4
    ('tukey_time spent', CustomTukeyTransformer('Time Spent', 'inner')),  #from chapter 4
    ('scale_age', CustomRobustTransformer('Age')), #from 5
    ('scale_time spent', CustomRobustTransformer('Time Spent')), #from 5
    ('impute', KNNImputer(n_neighbors=5, weights="uniform", add_indicator=False)),
    ], verbose=True)

########################################################################################################################################################

#fitted_pipeline = titanic_transformer.fit(X_train, y_train)  #notice just fit method called
#joblib.dump(fitted_pipeline, 'fitted_pipeline.pkl')

########################################################################################################################################################
# CHAPTER 9 DATA SETUP FUNCTIONS 
########################################################################################################################################################
def dataset_setup(original_table, label_column_name:str, the_transformer, rs, ts=.2):
  table_features = original_table.drop(columns=label_column_name)
  labels = original_table[label_column_name].to_list()
  X_train, X_test, y_train, y_test = train_test_split(table_features, labels, test_size=ts, shuffle=True,
                                                      random_state=rs, stratify=labels)
  
  X_train_transformed = the_transformer.fit_transform(X_train, y_train)
  X_test_transformed = the_transformer.transform(X_test)

  x_train_numpy = X_train_transformed.to_numpy()
  x_test_numpy = X_test_transformed.to_numpy()
  y_train_numpy = np.array(y_train)
  y_test_numpy = np.array(y_test)
  return x_train_numpy, x_test_numpy, y_train_numpy, y_test_numpy


def titanic_setup(titanic_table, transformer=titanic_transformer, rs=titanic_variance_based_split, ts=.2):
  return dataset_setup(titanic_table, 'Survived', transformer, rs=rs, ts=ts)


def customer_setup(customer_table, transformer=customer_transformer, rs=customer_variance_based_split, ts=.2):
  return dataset_setup(customer_table, 'Rating', transformer, rs=rs, ts=ts)
