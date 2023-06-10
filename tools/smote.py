from imblearn.over_sampling import SMOTE

def smote(args,X,y):
    """
    SMOTE - Synthetic Minority Over-sampling Technique
    That is, for each minority class sample, we choose its nearest minority class neighbor,
        Args:
            args: arguments from argument_parser()
            X: features
            y: labels
        
        Returns:
            X_resampled: resampled features
            y_resampled: resampled labels
    """
    
    smote = SMOTE(random_state=args.seed)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return X_resampled, y_resampled

