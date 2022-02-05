# data_skew_impact

A complete pipeline for 
Investigation of data skew on regression problem using multiple kaggle datasets.

Base condition for model training are:
- Nan processing (columns with more than 50% of nan values were neglected, mean value fill for continuous columns, most occuring fill for categorical variables).
- Normalization / standardization of continuous data
- Onehot encoding of categorical variables ().

Following techniques were employed to detect any impact of data skew on model performances:
- Feature selection at 0.5 threshold on continuous columns only after onehot encoding.
- Feature selection at 0.5 threshold on all columns after encoding (normally not recommended but tried for sake of the method).
- Hyperparameter optimization
- 5 fold cross validation

The seed was set to be the same for all experiment so the data content was the same. Just the methods changed.
Even if some method did not make sense (feature selection on categorical cols, they were tried).
Overfitting was prevented using early stopping

Data
https://www.kaggle.com/c/house-prices-advanced-regression-techniques
https://www.kaggle.com/c/nyc-taxi-trip-duration
https://www.kaggle.com/c/new-york-city-taxi-fare-prediction
https://www.kaggle.com/c/how-much-did-it-rain-ii
