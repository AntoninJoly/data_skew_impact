import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

class SkewAnalysis():
    def __init__(self):
        self.col = 'default' # continuous = int/float; categorical = object
        self.method_col = {'continuous': 'mean', # Replace Nan by mean value
                           'categorical':'most'} # Replace Nan value by most occuring
        self.df = None
        self.corr = None
        self.df_unskew = None
        self.skewness = None

        self.target_col = None
        self.continuous_corr_cols = None

        self.norm = None
        self.encoder = None

        self.transformation = ['log','squared','exp','square','cube','fourth']
    
    def handle_nan(self, df, target_col, col = None, method_col = None):
        if col == None:
            col = self.col
        if method_col == None:
            method_col = self.method_col
        
        self.target_col = target_col

        # assert col
        nan = pd.DataFrame(df.isna().sum()).T
        keep = [i for i in nan.columns if nan.loc[0,i] <= 0.5*df.shape[0]]
        drop = [i for i in nan.columns if nan.loc[0,i] > 0.5*df.shape[0]]
        
        df = df[keep]
        nan_col = df.columns[df.isna().any()].tolist()
        
        print('Following columns were remove because they contain more than 50% of Nan values :', drop)
        for c in nan_col:
            if df.loc[:,c].dtype == 'object': # Replace Nan by most occuring if categorical
                df[c].fillna(df[c].mode()[0])
            else: # Replace Nan by mean
                df[c].fillna((df[c].mean()), inplace=True)
        self.df = df
        self.df_unskew = df
        return nan.loc[:,drop], df

    def visualize_correlations(self):
        corr = self.df.loc[:,self.corr_col]
        # Visualization
        sns.set(font_scale=1.5)
        plt.figure(figsize=(20,20))
        with sns.axes_style("white"):
            ax = sns.heatmap(corr.corr().loc[[self.target_col],:],
                             vmax=.3,
                             square=True,
                             annot=True,
                             cbar_kws={"orientation": "horizontal"})
        
    def correlate_target(self, ratio=0.5):
        correlation = self.df.corr()
        self.corr_col = list(correlation[self.target_col][correlation[self.target_col] > ratio].index)
        self.corr = self.df.loc[:,self.corr_col]
        self.continuous_corr_cols = [i for i in self.corr.columns if self.corr.loc[:,i].dtype!='object']

    def unskew_correlated_variables(self):
        def process_data(data):
            for i in data.columns:
                method = self.transform[i]
                if method == 'Log':
                    data[i] = np.log1p(data[i])
                elif method == 'Root':
                    data[i] = np.sqrt(data[i])
            return data

        self.correlate_target()
        df = self.df[self.continuous_corr_cols]
        # Apply reversible transformation to dataframe
        df_log = pd.DataFrame(np.log1p(df.copy()), columns= self.continuous_corr_cols)
        df_square = pd.DataFrame(np.sqrt(df.copy()), columns= self.continuous_corr_cols)

        # Compute skewness of each dataframe
        skewed = df.skew()
        skewed_log = df_log.skew()
        skewed_square = df_square.skew()

        self.skewness = pd.concat([skewed, skewed_log, skewed_square], axis=1)
        self.skewness.columns = ['Raw','Log','Root']
        self.transform = self.skewness.abs().idxmin(axis=1).to_dict()
                
        unskew = process_data(df)

        # Replace unskewed column in original dataframe
        for idx, c in enumerate(unskew.columns):
            self.df_unskew[c] = unskew[c]
        skew_dict = {'original':skewed.to_dict(),
                     'unskewed':unskew.skew().to_dict(),
                     'transformation':self.transform}
        return self.df_unskew, skew_dict

    def visualize_skew(self, corr):
        ax = self.skewness.abs().plot.bar(rot=90,figsize=(10,7))

    def normalize_and_encode(self, keep_corr):
        continuous_col = [i for i in self.df.columns if self.df.loc[:,i].dtype!='object']
        categorical_col = [i for i in self.df.columns if self.df.loc[:,i].dtype=='object']

        df_continuous = self.normalize_continuous(continuous_col)
        df_categorical = self.encode_categorical(categorical_col)

        self.df = pd.concat((df_continuous, df_categorical),axis=1)

        if keep_corr:
            self.correlate_target()
            self.df = self.corr

        return self.df, df_continuous, df_categorical

    def normalize_continuous(self, col, method='mean'):
        if method == 'mean':# Mean normalization
            mean = self.df_unskew.loc[:,col].mean()
            std = self.df_unskew.std()
            self.norm = {'mean':mean,'std':std}
            self.df_unskew.loc[:,col] = (self.df_unskew.loc[:,col]-mean)/std
        elif method == 'minmax':# Min-max normalization
            min_ = self.df_unskew.loc[:,col].min()
            max_ = self.df_unskew.loc[:,col].max()
            self.norm = {'min':min_,'max':max_}
            self.df_unskew.loc[:,col] = (self.df_unskew.loc[:,col]-min_)/(max_ - min_)
        return self.df_unskew.loc[:,col]

    def encode_categorical(self, col, method='onehot'):
        
        self.encoder = OneHotEncoder(dtype=np.int, sparse=True)
        df_categorical = pd.DataFrame(self.encoder.fit_transform(self.df.loc[:,col]).toarray(),
                                      columns = [i for i in self.encoder.get_feature_names()],
                                      index = self.df.index)
        return df_categorical

    def encode(self):
        return []

    def decode(self, pred, y):
        index = y.index
        y = y.to_numpy()
        if list(self.norm.keys()) == ['mean','std']:
            mean = self.norm['mean'][self.target_col]
            std = self.norm['std'][self.target_col]
            pred = (pred * std) + mean
            y = (y * std) + mean
        tr = self.transform[self.target_col]
        if tr == 'Log':
            pred, y = np.expm1(pred), np.expm1(y) # Inverse log - Exp
        elif tr == 'Root':
            pred, y = np.square(pred), np.square(y) # Inverse root - square
        return pred, y, index

