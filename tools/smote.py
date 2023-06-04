from imblearn.over_sampling import SMOTE

def smote(X,y):

    # using SMOTE
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # # update dataframe
    # scale_damage_resampled = pd.Series(y_resampled)
    # data_resampled = X_resampled.copy()
    # data_resampled['scale_damage'] = scale_damage_resampled
    
    return X_resampled, y_resampled

