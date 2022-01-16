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
        self.skewness = None

        self.target_col = None
        
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
        
        return nan.loc[:,drop], df

    def visualize_correlations(self, df):
        corr = df.loc[:,self.corr_col]
        # Visualization
        sns.set(font_scale=1.5)
        plt.figure(figsize=(20,20))
        with sns.axes_style("white"):
            ax = sns.heatmap(corr.corr().loc[[self.target_col],:],
                             vmax=.3,
                             square=True,
                             annot=True,
                             cbar_kws={"orientation": "horizontal"})
        
    def feature_selection(self, df, col=None, ratio=0.5):
        if col == None:
            continuous_col = [i for i in df.columns if df.loc[:,i].dtype!='object']
            categorical_col = [i for i in df.columns if df.loc[:,i].dtype=='object']
            correlation = df[continuous_col].corr()
            corr_col = list(correlation[self.target_col][correlation[self.target_col] > ratio].index)
            return df.loc[:, corr_col + categorical_col]
        else:
            correlation = df[col].corr()
            corr_col = list(correlation[self.target_col][correlation[self.target_col] > ratio].index)
            other_col = [i for i in df.columns if i not in col]
            return df.loc[:, corr_col + other_col]

    def unskew(self, df):
        def process_data(data):
            for i in data.columns:
                method = self.transform[i]
                if method == 'Log':
                    data[i] = np.log1p(data[i])
                elif method == 'Root':
                    data[i] = np.sqrt(data[i])
            return data

        continuous_col = [i for i in df.columns if df.loc[:,i].dtype!='object']
        df_continuous = df.loc[:,continuous_col]

        # Apply reversible transformation to dataframe
        df_log = pd.DataFrame(np.log1p(df_continuous.copy()), columns= df_continuous.columns)
        df_square = pd.DataFrame(np.sqrt(df_continuous.copy()), columns= df_continuous.columns)

        # Compute skewness of each dataframe
        skewed = df_continuous.skew()
        skewed_log = df_log.skew()
        skewed_square = df_square.skew()

        self.skewness = pd.concat([skewed, skewed_log, skewed_square], axis=1)
        self.skewness.columns = ['Raw','Log','Root']
        self.transform = self.skewness.abs().idxmin(axis=1).to_dict()
                
        unskew = process_data(df_continuous)
        # Replace unskewed column in original dataframe
        for idx, c in enumerate(unskew.columns):
            df[c] = unskew[c]
        skew_dict = {'original':skewed.to_dict(),
                     'unskewed':unskew.skew().to_dict(),
                     'transformation':self.transform}

        return df, skew_dict

    def visualize_skew(self, drop_non_corr=True):
        ax = self.skewness.abs().plot.bar(rot=90,figsize=(10,7))

    def normalize_and_encode(self, df):
        continuous_col = [i for i in df.columns if df.loc[:,i].dtype!='object']
        categorical_col = [i for i in df.columns if df.loc[:,i].dtype=='object']

        df_continuous = self.normalize_continuous(df.loc[:,continuous_col])
        df_categorical = self.encode_categorical(df.loc[:,categorical_col])

        df = pd.concat((df_continuous, df_categorical), axis=1)

        return df

    def normalize_continuous(self, df, method='mean'):
        if method == 'mean':# Mean normalization
            mean = df.mean()
            std = df.std()
            self.norm = {'mean':mean,'std':std}
            df = (df-mean)/std
        elif method == 'minmax':# Min-max normalization
            min_ = df.loc[:,col].min()
            max_ = df.loc[:,col].max()
            self.norm = {'min':min_,'max':max_}
            df = (df-min_)/(max_ - min_)
        return df

    def encode_categorical(self, df, method='onehot'):
        self.encoder = OneHotEncoder(dtype=np.int, sparse=True)
        df_categorical = pd.DataFrame(self.encoder.fit_transform(df).toarray(),
                                      columns = [i for i in self.encoder.get_feature_names()],
                                      index = df.index)
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

