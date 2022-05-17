import numpy as np

#inspired by https://github.com/pwesp/random-forest-polyp-classification, 

def feature_selection(df, corr_threshold=0.8):
    print("Remove correlated features")

    # Check corr_threshold
    if type(corr_threshold) is float:
        if 0 < corr_threshold and corr_threshold < 1:
            print("Correlation threshold:", corr_threshold)
        else:
            print("ERROR: Threshold must be a float value between 0.0 and 1.0.")
            return -1
    else:
        print("ERROR: Threshold must be a float value between 0.0 and 1.0.")
        return -1
    
    # Get all features
    all_features = df.columns.to_list()
    n_features   = df.shape[1]

    # Compute correlation matrix
    corr = df.corr(method='pearson')
    corr = corr.abs()

    # Keep only correlation values in the upper traingle matrix
    triu = np.triu(np.ones(corr.shape), k=1)
    triu = triu.astype(bool)
    corr = corr.where(triu)
    
    # Select columns which will be dropped from dataframe
    cols_to_drop = [column for column in corr.columns if any(corr[column] > corr_threshold)]

    n_cols_to_drop = len(cols_to_drop)
    p_cols_to_drop = 100 * (float(n_cols_to_drop) / float(n_features))
    p_cols_to_drop = np.round(p_cols_to_drop, decimals=1)
    print("Drop", n_cols_to_drop, "/", n_features, " features (", p_cols_to_drop, "%).")

    # Drop colums
    df = df.drop(cols_to_drop, axis=1)
    
    # Find names of features which have been dropped
    uncorrelated_features = df.columns.to_list()
    dropped_features      = list(set(all_features) - set(uncorrelated_features))
    
    return df, dropped_features