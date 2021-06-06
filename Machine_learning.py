import numpy as np
import sklearn

def VarianceOnRMSE(X, y, estimator, n_bootstrap, n_train, K_fold=False, refit=False, random_state=0, verbose=False):
    """
    Create bootstrap samples from the original dataset and train an estimator for each of those samples. 
    The mean and variance is returned

    Parameters
    ----------
    X,y: numpy arrays
        features and outcome, must be the same length
    estimator: sklearn estimator
        needs to be fitted to data already when refit is false
    n_bootstrap: int 
        number of bootstrap iterations or folds
    n_train: int or float
        if int, the number of samples that should be in the training set (must be < total amount of samples)
        if float, the proportion of data that should be in the training set (must be between 0 and 1)
    K_fold: bool, default=False
        use k-fold cross validation instead of bootstrap sampling. number of folds is determined with n_bootstrap
    refit: bool, default=False
        if true, refit the estimator on each of the bootstrap samples and tests on the rest
        if false, skip the refit step and immediately tests the algorithm
    random_state: int or RandomState or None, default=0
        seed for the RNG, usefull for creating reproducible results
    verbose: bool, default=False
        if True, prints status messages and additional outputs

    Returns
    -------
    avg, std: float
        average and standard deviation of all bootstrap rmse's
    """

    print("Determining average RMSE and its variance...") if verbose else None
    rmse_list = []
    if K_fold:
        bs = sklearn.model_selection.KFold(n_bootstrap, shuffle=True, random_state=random_state)
    else:
        bs = sklearn.model_selection.ShuffleSplit(n_bootstrap, train_size=n_train, random_state=random_state)
        

    for train_index, test_index in bs.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        if refit:
            estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_test)
        rmse = sklearn.metrics.mean_squared_error(y_test, y_pred, squared = False)
        rmse_list.append(rmse)

    avg = np.average(rmse_list)
    std = np.std(rmse_list)
    print("Average and variance determined!") if verbose else None
    return avg, std


def GoodFeatureSelector(X, imp, threshold=0.0, verbose=False):
    """
    function to use in conjunction with rfpimp's importances function to create a dataset where only
    usefull features are left over
    
    Paramters
    ---------
    X: Pandas dataframe
        dataframe containing the features
    imp: Pandas dataframe
        dataframe of the feature importances. not necessarily sorted
    threshold: float, default=0.0
        threshold for "usefull" features. every feature with less importance is removed
    verbose: bool, default=False
        if True, prints status messages and additional outputs

    Returns
    -------

    """
    print("Removing unimportant features...") if verbose else False
    feat_len = len(imp["Importance"])
    keep_list = []
    for i, rows in imp.iterrows():
        if rows["Importance"] > threshold:
            keep_list.append(i)

    if len(keep_list) == feat_len:
        no_change = True
    else:
        no_change = False
    X = X.loc[:, keep_list]

    print("{} unimportant features removed!".format(feat_len - len(keep_list))) if verbose else False
    return X, keep_list, no_change