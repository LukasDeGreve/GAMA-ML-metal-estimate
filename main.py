"""
Code to estimate metallicity of GAMA galaxies with machine learning
Is able to produce the figures and stuff from chapter 4, but not all at once
Most important is probably around line 40, to set all options of the program
Some legacy stuff with crossmatching and comparing with acquaviva is still included, but commented out
"""

import numpy as np
from numpy.lib.shape_base import split
from scipy.stats.stats import scoreatpercentile
import sklearn
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn import svm, linear_model, ensemble
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
import pandas as pd
import matplotlib.pyplot as plt
import Plots
import Dataprocessing as dp
import Machine_learning as ml
from rfpimp import *
import time

#* Importing data and defining some stuff
bands_list = ["FUV_flux", "NUV_flux", "u_flux", "g_flux", "r_flux", "i_flux", "z_flux", "X_flux", "Y_flux", "J_flux", "H_flux", "K_flux","W1_flux", "W2_flux", "W3_flux", "W4_flux", "P100_flux", "P160_flux", "S250_flux", "S350_flux", "S500_flux"]
SDSS_list = ["u_flux", "g_flux", "r_flux", "i_flux", "z_flux"]
WISE_list = ["W1_flux", "W2_flux", "W3_flux", "W4_flux"]
Herschel_list = ["P100_flux", "P160_flux", "S250_flux", "S350_flux", "S500_flux"]
GALEX_list = ["FUV_flux", "NUV_flux"]
VIKING_list = ["X_flux", "Y_flux", "J_flux", "H_flux", "K_flux"]
custom_list = ["g_flux", "i_flux"]

#* program options
z_low = 0.07                #* redshift lower bound (0.07, 0.18)
z_high = 0.11               #* redshift upper bound (0.11, 0.25)
use_mass = False            #* use mass as a feature
use_redshift = True         #* use redshift as a feature
grid_search = True          #* perform a gridsearch to find the best model
feature_select = True       #* use feature selection to remove "useless" features and reduce overfitting
only_tree = True            #* Only use extremely random forests, found to be the best estimator type
n_splits = 4                #* number of splits, is the same for both the internal and external loop
dataprocess = False         #* rerun dataprocessing
use_cigale = False           #* uses the CIGALE fitted data instead of the GAMA data
bands_list = bands_list     #* change this to quickly change which bands are used
use_colours = True          #* add colours to the list of features
prediction = "METAL"        #* What the algorithm predicts. "METAL", "MASS"
scoring="neg_mean_squared_error"  

#* Data processing
#* this takes quite a while to do every time, everything in if is collected in the merged dataframe
if dataprocess:
    simple = pd.read_csv("/run/media/lukasdg/Seagate/Documents/School/2e master/Thesis/Data/GaussFitSimple.csv")
    complex = pd.read_csv("/run/media/lukasdg/Seagate/Documents/School/2e master/Thesis/Data/GaussFitComplex.csv")
    bands = pd.read_csv("/run/media/lukasdg/Seagate/Documents/School/2e master/Thesis/Data/LambdarCat.csv")
    mass = pd.read_csv("/run/media/lukasdg/Seagate/Documents/School/2e master/Thesis/Data/StellarMassesLambdar.csv")

    print(mass.shape)
    mass = dp.AddMass(mass, verbose=True)
    print(mass.shape)

    lines = simple.merge(complex, on=["CATAID", "SPECID"])
    print(lines.shape)
    # always use KeepBestGalaxy first, before removing stuff with other functions
    lines = dp.KeepBestGalaxy(lines, verbose=True)
    print(lines.shape)
    lines = dp.SelectSurvey(lines, ("GAMA", "SDSS"), verbose=True)
    print(lines.shape)
    lines = dp.RemoveBadFit(lines, method="PG16", verbose=True)
    print(lines.shape)
    lines = dp.AddMetallicity(lines, method="PG16", verbose=True)
    print(lines.shape)
    lines = dp.RemoveAGN(lines, verbose=True)
    print(lines.shape)

    print(bands.shape)
    bands = dp.RemoveMissing(bands, bands_list, verbose=True)
    print(bands.shape)
    bands = dp.FluxToMagnitude(bands, bands_list, verbose=True)

    master = bands.merge(lines[["CATAID", "METAL", "Z"]], on="CATAID")
    print(master.shape)
    master = master.merge(mass[["CATAID", "MASS"]], on="CATAID")
    print(master.shape)
    master.to_csv("all_bands_metals.csv", index=False)
    # all galaxies with mass have an SFR as well, but it is not added here as it won't be used as a feature

if not dataprocess:
    merged = pd.read_csv("/run/media/lukasdg/Seagate/Documents/School/2e master/Thesis/Python_scripts/all_bands_metals.csv")
    master = merged
    if use_colours and not use_cigale:
        master, extra_colours = dp.AddColours(master, bands_list, neighbours=21, squared=False, verbose=True)

    if use_cigale:
        cigale = dp.PrepCigaleFile("Cigale/resultz.csv", verbose=True)
        cigale = dp.FluxToMagnitude(cigale, bands_list, verbose=True)

        if use_colours:
            cigale, extra_colours = dp.AddColours(cigale, bands_list, neighbours=21, squared=False, verbose=True)    
    

#* SDSS stuff was needed for previous versions, but is left for completeness. Most of it can be commented out for a better performance and less memory usage. Not sure if everything still works
#* merge sdss_info and sdss_mets if Acquaviva metallicities are required
#sdss_info =  pd.read_csv("/home/lukasdg/Documents/External_documents/School/2e master/Thesis/Acquaviva 15/gal_info_dr7_v5_2.csv")
#sdss_mets = pd.read_csv("/home/lukasdg/Documents/External_documents/School/2e master/Thesis/Acquaviva 15/gal_fiboh_dr7_v5_2.csv")
#sdss_info = pd.merge(sdss_info, sdss_mets, left_index=True, right_index=True)

#* at this index, RA and DEC are -9999, so we will drop this point. Only relevant if working with acquaviva data
#sdss_info = sdss_info.drop([170713])

#* uncomment to crossmatch data with the galaxies Acquaviva used
#master = dp.Crossmatch(master, sdss_info, verbose=1)

if not use_cigale:
    master = dp.SelectRedshift(master, z_low, z_high, verbose=1)
else: 
    cigale = dp.SelectRedshift(cigale, z_low, z_high, verbose=1)
    del_list=[]
    print(cigale.shape)
    for i, rows in cigale.iterrows():
        if rows["METAL"] > 9.25:
            del_list.append(i)
    cigale = cigale.drop(del_list)
    print(cigale.shape)

#* Machine learning
if use_mass:
    bands_list.append("MASS")
if use_redshift:
    bands_list.append("Z")
if use_colours:
    bands_list.extend(extra_colours)
if not use_cigale:
    X = master[bands_list].fillna(0).values
    y = master[prediction].values
else:
    X = cigale[bands_list].fillna(0).values
    y = cigale[prediction].values


#* split in 4 folds and use each fold once as test set.
#* First fold is used for hyperparameter optimization as well
#* Root of the average of the mean squared errors of each fold is used as the reported rmse. The error is stddev of root of those 4 mse's.
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
rme = []
r2 = []
scaler = StandardScaler()
kf = KFold(n_splits = n_splits)

bad_features = True      #! Don't change this! 
if not grid_search:
    best_model = ensemble.ExtraTreesRegressor(min_samples_leaf=1, min_samples_split=8, n_estimators=50)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)
    avg_rmse, rmse_err = ml.VarianceOnRMSE(X_train, y_train, best_model, 10, 0.75, K_fold=True, refit=True, verbose=True)
    y_pred = best_model.predict(X_test)
    avg_r2 = sklearn.metrics.r2_score(y_test, y_pred)
    r2_err = 0

else:
    splits = sklearn.model_selection.KFold(n_splits, shuffle=True, random_state=1)
    for train_index, test_index in splits.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        

        if grid_search:
            while bad_features:
                if only_tree:
                    # Extremely random trees
                    print("Extremely random trees")
                    rtree = ensemble.ExtraTreesRegressor(random_state=1)
                    rtree_params = [{'n_estimators':[10, 25, 50, 100], 'min_samples_split':[2,4,6,8,10], 'min_samples_leaf':[1, 2, 4, 6, 8, 10]}]

                    rtree_gs = GridSearchCV(rtree, rtree_params, scoring=scoring, cv=kf, n_jobs=-1, verbose=1)
                    rtree_gs.fit(X_train, y_train)
                    score = np.sqrt(-1*rtree_gs.best_score_)

                    print(score)
                    print(rtree_gs.best_params_)
                    best_score = score
                    best_model = rtree_gs.best_estimator_
                    print()

                else:
                    # Ridge regression
                    print("Ridge regression")
                    ridge = linear_model.Ridge()
                    ridge_params = [{'alpha':param_range}]

                    tic = time.perf_counter()
                    ridge_gs = GridSearchCV(ridge, ridge_params, scoring=scoring, cv=kf, n_jobs=-1, verbose=1)
                    ridge_gs.fit(X_train, y_train)
                    toc = time.perf_counter()
                    score = np.sqrt(-1*ridge_gs.best_score_)

                    print("rmse: {}".format(score))
                    print("R2: {}".format(sklearn.metrics.r2_score(y_test, ridge_gs.predict(X_test))))
                    print("time elapsed: {:.4f} seconds".format(toc-tic))
                    print(ridge_gs.best_params_)
                    best_score = score
                    best_model = ridge_gs.best_estimator_
                    print()


                    # Random forest
                    print("Random forest")
                    forest = ensemble.RandomForestRegressor(random_state=1)
                    forest_params = [{'n_estimators':[10, 25, 50, 100], 'min_samples_split':[2,4,6,8,10], 'min_samples_leaf':[1, 2, 4, 6, 8, 10]}]

                    tic = time.perf_counter()
                    forest_gs = GridSearchCV(forest, forest_params, scoring=scoring, cv=kf, n_jobs=-1, verbose=1)
                    forest_gs.fit(X_train, y_train)
                    toc = time.perf_counter()
                    score = np.sqrt(-1*forest_gs.best_score_)

                    print("rmse: {}".format(score))
                    print("R2: {}".format(sklearn.metrics.r2_score(y_test, forest_gs.predict(X_test))))
                    print("time elapsed: {:.4f} seconds".format(toc-tic))
                    print(forest_gs.best_params_)
                    if score < best_score:
                        best_score = score
                        best_model = forest_gs.best_estimator_
                    print()

                    '''
                    # Extremely random trees
                    print("Extremely random trees")
                    rtree = ensemble.ExtraTreesRegressor()
                    rtree_params = [{'n_estimators':[10, 25, 50, 100], 'min_samples_split':[2,4,6,8,10], 'min_samples_leaf':[1, 2, 4, 6, 8, 10]}]

                    tic = time.perf_counter()
                    rtree_gs = GridSearchCV(rtree, rtree_params, scoring=scoring, cv=kf, n_jobs=-1, verbose=1)
                    rtree_gs.fit(X_train, y_train)
                    toc = time.perf_counter()
                    score = np.sqrt(-1*rtree_gs.best_score_)

                    print("rmse: {}".format(score))
                    print("R2: {}".format(sklearn.metrics.r2_score(y_test, rtree_gs.predict(X_test))))
                    print("time elapsed: {:.4f} seconds".format(toc-tic))
                    print(rtree_gs.best_params_)
                    if score < best_score:
                        best_score = score
                        best_model = rtree_gs.best_estimator_
                    print()
                    '''

                    # Adaboost
                    print("adaboost")
                    ada = ensemble.AdaBoostRegressor()
                    ada_params = [{'base_estimator':[DecisionTreeRegressor(max_depth=2), DecisionTreeRegressor(max_depth=4), DecisionTreeRegressor(max_depth=6), DecisionTreeRegressor(max_depth=8), DecisionTreeRegressor(max_depth=10)], 'n_estimators':[10, 25, 50, 100], 'loss':['linear', 'square', 'exponential']}]

                    tic = time.perf_counter()
                    ada_gs = GridSearchCV(ada, ada_params, scoring=scoring, cv=kf, n_jobs=-1, verbose=1)
                    ada_gs.fit(X_train, y_train)
                    toc =  time.perf_counter()
                    score = np.sqrt(-1*ada_gs.best_score_)

                    print("rmse: {}".format(score))
                    print("R2: {}".format(sklearn.metrics.r2_score(y_test, ada_gs.predict(X_test))))
                    print("time elapsed: {:.4f} seconds".format(toc-tic))
                    print(ada_gs.best_params_)
                    if score < best_score:
                        best_score = score
                        best_model = ada_gs.best_estimator_
                    print()

                    
                    # SVM 
                    print("SVM")
                    sv = svm.SVR()
                    sv_params = [{'kernel':['linear', 'rbf'], 'gamma':param_range, 'C':param_range}]
                    
                    tic = time.perf_counter()
                    sv_gs = GridSearchCV(sv, sv_params, scoring=scoring, cv=kf, n_jobs=-1, verbose=1)
                    sv_gs.fit(X_train, y_train)
                    toc = time.perf_counter()
                    score = np.sqrt(-1*sv_gs.best_score_)

                    print("rmse: {}".format(score))
                    print("R2: {}".format(sklearn.metrics.r2_score(y_test, sv_gs.predict(X_test))))
                    print("time elapsed: {:.4f} seconds".format(toc-tic))
                    print(sv_gs.best_params_)
                    if score < best_score:
                        best_score = score
                        best_model = sv_gs.best_estimator_
                    print()
                
                print("best score: {}".format(best_score))
                print("best model: {}".format(best_model))

                # the importances function need dataframes instead of numpy arrays
                
                X_train = pd.DataFrame(X_train, columns=bands_list)
                X_test = pd.DataFrame(X_test, columns=bands_list)
                y_train = pd.Series(y_train, name="Z")

                imp = importances(best_model, X_train, y_train)
                viz = plot_importances(imp, imp_range=(-0.001, imp.iloc[0].tolist()[0]*1.1))

                if feature_select:
                    X_train, bands_list, no_change = ml.GoodFeatureSelector(X_train, imp, 0.001, verbose=True)

                X_train = X_train.to_numpy()
                X_test = X_test.loc[:, bands_list]
                X_test = X_test.to_numpy()

                if not feature_select or no_change:
                    bad_features = False

                #* code snippet to only use best x features instead of using a cutoff with GoodFeatureSelector. Only works if the importance list is ordered! Also make sure you retrain the algorithm before estimating rmse (can be done by making sure it goes in the loop again aka high enough threshold on feature importance)
                '''
                X_train = pd.DataFrame(X_train, columns=bands_list)
                X_test = pd.DataFrame(X_test, columns=bands_list)
                bands_list = bands_list[:1]
                print("Lines used: {}".format(bands_list))
                X_train = X_train.loc[:, bands_list]
                X_test = X_test.loc[:, bands_list]
                X_train = X_train.to_numpy()
                X_test = X_test.to_numpy()
                '''     
        

        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        rme.append(sklearn.metrics.mean_squared_error(y_test, y_pred, squared=True))
        r2.append(sklearn.metrics.r2_score(y_test, y_pred))

    #* get average and error of the metrics
    #* regular average instead of weighted is taken as the difference in samples between the 4 folds is 1 at max
    avg_rmse = np.sqrt(np.average(rme))
    rmse_err = np.std(np.sqrt(rme))
    avg_r2 = np.average(r2)
    r2_err = np.std(r2)
    print(np.sqrt(rme))
    print(r2)

#* Done with the grid search. Plot some results:
colours = Plots.set_defaults()

#* plot true versus predicted metallicity/mass of last fold
plt.figure(4)
x, y, c = Plots.estimate_density(y_test, y_pred)
plt.scatter(x, y, c=c, alpha=0.8)
box = dict(facecolor='none', edgecolor='lightgray', boxstyle='round')
if prediction == "METAL":
    plt.ylabel("Predicted metallicities")
    plt.xlabel("True metallicities")
    plt.xlim(7.95, 9.25)
    plt.ylim(7.95, 9.25)
    x_line = np.linspace(7.9, 9.3, 50)  
    plt.text(8.8, 8.1, r'rmse: {:.3f} $\pm$ {:.3f}' '\n' r'$R^2$' ': {:.3f} $\pm$ {:.3f}'.format(avg_rmse, rmse_err, avg_r2, r2_err), bbox=box)

elif prediction == "MASS":
    plt.ylabel("Predicted masses")
    plt.xlabel("True masses")
    plt.xlim(8.35, 11.65)
    plt.ylim(8.25, 11.75)
    x_line = np.linspace(8.0, 12, 50)
    plt.text(10.5, 8.8, r'rmse: {:.3f} $\pm$ {:.3f}' '\n' r'$R^2$' ': {:.3f} $\pm$ {:.3f}'.format(avg_rmse, rmse_err, avg_r2, r2_err), bbox=box)


plt.plot(x_line, x_line)
plt.plot(x_line, x_line + 0.2, linestyle='dotted', color=colours[1])
plt.plot(x_line, x_line - 0.2, linestyle='dotted', color=colours[1])
plt.title("all bands, low z bin")
#plt.plot(x_line, np.array([8.3 for i in range(len(x_line))]), linestyle='dashed', color=colours[2])
#plt.plot(x_line, np.array([9.07 for i in range(len(x_line))]), linestyle='dashed', color=colours[2])
plt.figure(5)
#n, bins, patches = plt.hist(y_test, bins=20, rwidth=0.95, alpha=1, label="True metallicity", color=colours[4])
#plt.hist(y_pred, bins, rwidth=0.95, alpha=0.6, label="Predicted metallicity", color=colours[0])
plt.hist([y_pred, y_test], bins=20, rwidth=0.85, label=["predicted", "true"], color=[colours[0], colours[3]])
plt.legend(loc='upper left')


#* learning curve
plt.figure(3)
train_sizes, train_scores, test_scores = learning_curve(estimator=best_model, X=X_train, y=y_train, train_sizes=np.linspace(0.05, 1, 20), cv=10, n_jobs=-1, scoring=scoring)
train_mean = np.sqrt(-1*np.mean(train_scores, axis=1))
train_std = np.std(train_scores, axis=1)
test_mean = np.sqrt(-1*np.mean(test_scores, axis=1))
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color=colours[2], marker='o', markersize=5, label='training RMSE')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color=colours[2])
plt.plot(train_sizes, test_mean, color=colours[5], linestyle='--', marker='s', markersize=5, label='validation RMSE')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color=colours[5])
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('RMSE')
plt.legend(loc='upper right')
plt.title("large sample")
plt.show()

