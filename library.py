import pandas as pd
import numpy as np
import sklearn
sklearn.set_config(transform_output="pandas")  #says pass pandas tables through pipeline instead of numpy matrices
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


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

  #write the transform method with asserts. Again, maybe copy and paste from MappingTransformer and fix up.
  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'

    # check if all columns exist and retrieve any bogus column names
    mapping_keys = set(self.mapping_dict.keys())
    x_keys = set(X.columns.to_list())
    keys_not_found = mapping_keys - x_keys

    assert len(keys_not_found) == 0, f'{self.__class__.__name__}.transform unknown column(s) "{keys_not_found}"'  #columns legit?

    #do actual mapping
    X_ = X.copy()
    X_.rename(columns=self.mapping_dict, inplace=True)
    return X_


  #write fit_transform that skips fit
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

  #fill in the rest below
  def fit(self, X, y = None):
    print(f"\nWarning: {self.__class__.__name__}.fit does nothing.\n")
    return self

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'{self.__class__.__name__}.transform expected Dataframe but got {type(X)} instead.'
    assert self.target_column in X.columns.to_list(), f'{self.__class__.__name__}.transform unknown column "{self.target_column}"'  #column legit?

# A bit confused when to wrangle NaN, the transformer seems to work without this code
    #placeholder = "NaN"
    #column_values = X[self.target_column].fillna(placeholder).tolist()  #convert all nan values to the string "NaN" in new list
    #column_values = [np.nan if v == placeholder else v for v in column_values]  #now convert back to np.nan

    #do actual mapping
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
