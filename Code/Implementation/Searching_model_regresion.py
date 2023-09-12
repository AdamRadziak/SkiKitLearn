import joblib
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from Code.Libraries.Datalibs import Data as dt

CSV_PATH = r"../../Datasets/gpa_study_hours.csv"
TEST_RATIO = 0.2
ID_COLUMN_CSV = "Id"

if __name__ == '__main__':
    # potok przekształcania danych
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler', StandardScaler()),
    ])
    # create data object
    data = dt(CSV_PATH)
    fetch_data = data.load_data()
    #
    # # cleaning the data from NA and add median to NA data
    for key in fetch_data:
        # if in column in data is null value replace with median
        if fetch_data[key].isnull().values.any() and isinstance(fetch_data[key], int):
            data.replacing_NA(fetch_data, key)
    # drop from fetch data gpa scores > 4 is not true
    for value in fetch_data["gpa"]:
        if value > 4.0:
            fetch_data.replace(value, 4.0, inplace=True)

    # all labels - desiresed scores
    all_labels = fetch_data["gpa"]
    # get test and train data from fetch_data
    train_set, test_set = train_test_split(fetch_data, train_size=0.2, random_state=20)

    # create copy of train_data numeric

    train_tr = num_pipeline.fit_transform(train_set)
    train_labels = train_set["gpa"]
    X_test = test_set.drop("gpa", axis=1)
    Y_test = test_set["gpa"]
    test_tr = num_pipeline.fit_transform(test_set)
    test_labels = test_set["gpa"]
    # data for checking the model
    some_data = fetch_data.iloc[:10]
    # # checking the model 5 first labels
    some_labels = all_labels.iloc[:10]
    # # transform data
    some_data_prepared = num_pipeline.fit_transform(some_data)

    # dostrajamy model metodą przeszukiwania siatki
    # n_estimators - ilosc drzew
    # max_features - liczba funkcji
    # bootstrap = False caly zbior danych uzywamy do budowy drzewa
    # model lasu losowego
    param_grid = [
        {'bootstrap': [True], 'n_estimators': [3, 10, 30, 40], 'max_features': [2, 4, 6, 8, 10]},
        {'bootstrap': [False], 'n_estimators': [3, 10, 30, 80, 100], 'max_features': [2, 5, 10, 15, 20]},
    ]
    forest_reg = RandomForestRegressor()

    grid_search_forest = RandomizedSearchCV(forest_reg, param_grid, cv=5,
                                            scoring='neg_mean_squared_error',
                                            return_train_score=True)
    # tworzymy model zgodnie z best params
    grid_search_forest.fit(train_tr, train_labels)
    print("najlepsze parametry model lasu: ", grid_search_forest.best_params_)
    # sprawdzian krzyzowy
    scores_forest = cross_val_score(forest_reg, train_tr, train_labels,
                                    scoring="neg_mean_squared_error", cv=10)
    data.display_scores(scores_forest, "Wyniki dla modelu lasu losowego:")
    data.saving_scores_to_file(scores_forest, custom_start_print="Wyniki dla modelu lasu losowego:\n")
    # etykiety
    print("Etykiety", list(some_labels))
    # prognozy
    print("Prognozy model losowego lasu:", grid_search_forest.predict(some_data_prepared))
    # model regresji liniowej
    lin_reg = LinearRegression()
    # tworzymy model zgodnie z best params
    lin_reg.fit(train_tr, train_labels)
    # sprawdzian krzyzowy
    scores_linear = cross_val_score(lin_reg, train_tr, train_labels,
                                    scoring="neg_mean_squared_error", cv=10)
    data.display_scores(scores_linear, "Wyniki dla modelu regresji liniowej:")
    data.saving_scores_to_file(scores_linear, custom_start_print="Wyniki dla modelu regresji liniowej:\n")
    # etykiety
    print("Etykiety", list(some_labels))
    # prognozy
    print("Prognozy model regresji liniowej:", lin_reg.predict(some_data_prepared))
    # model drzew decyzyjnych
    param_grid = [
        {'splitter': ['random'], 'max_features': [20, 24, 3, 40]},
        {'splitter': ['best'], 'max_leaf_nodes': [10, 20, 100, 120]},
    ]

    tree_reg = DecisionTreeRegressor()
    grid_search_tree = RandomizedSearchCV(tree_reg, param_grid, cv=5,
                                          scoring='neg_mean_squared_error',
                                          return_train_score=True)
    # tworzymy model zgodnie z best params
    grid_search_tree.fit(train_tr, train_labels)
    print("najlepsze parametry model drzew decyzyjnych: ", grid_search_tree.best_params_)
    # sprawdzian krzyzowy
    scores_tree = cross_val_score(tree_reg, train_tr, train_labels,
                                  scoring="neg_mean_squared_error", cv=10)
    data.display_scores(scores_tree, "Wyniki dla modelu drzew decyzjnych:")
    data.saving_scores_to_file(scores_tree, custom_start_print="Wyniki dla modelu drzew decyzjnych:\n")
    # etykiety
    print("Etykiety", list(some_labels))
    # prognozy
    print("Prognozy model drzew decyzyjnych:", grid_search_tree.predict(some_data_prepared))

    # linear predictions
    test_predict_lin = lin_reg.predict(test_tr)
    # mean squared error
    mean_sq_err = mean_squared_error(test_labels, test_predict_lin)
    rmse = np.sqrt(mean_sq_err)
    print("rmse regresja liniowa ", rmse)
    # tree predictions
    test_predict_tree = grid_search_tree.predict(test_tr)
    # mean squared error
    mean_sq_err = mean_squared_error(test_labels, test_predict_tree)
    rmse = np.sqrt(mean_sq_err)
    print("rmse drzewa decyzyjne ", rmse)
    # tree predictions
    test_predict_forest = grid_search_forest.predict(test_tr)
    # mean squared error
    mean_sq_err = mean_squared_error(test_labels, test_predict_forest)
    rmse = np.sqrt(mean_sq_err)
    print("rmse las losowy ", rmse)
    # model maszyny wektorów nośnych
    param_grid = [
        {'kernel': ['linear'], 'C': [1, 5, 10, 100]},
        {'kernel': ['rbf'], 'C': [1, 5, 10, 100], 'gamma': ['scale', 'auto', 2.5, 5.0]},
    ]
    svr_reg = SVR()

    grid_search_svr = RandomizedSearchCV(svr_reg, param_grid, cv=5,
                                         scoring='neg_mean_squared_error',
                                         return_train_score=True)
    # tworzymy model zgodnie z best params
    grid_search_svr.fit(train_tr, train_labels)
    print("najlepsze parametry model maszyny wektorów nośnych: ", grid_search_svr.best_params_)
    svr_reg_1 = grid_search_svr.best_estimator_
    # sprawdzian krzyzowy
    scores_svr = cross_val_score(svr_reg, train_tr, train_labels,
                                 scoring="neg_mean_squared_error", cv=10)
    data.display_scores(scores_svr, "Wyniki dla modelu maszyny wektorów nośnych:")
    data.saving_scores_to_file(scores_svr, custom_start_print="Wyniki dla modelu maszyny wektorów nośnych metoda 1:\n")
    # etykiety
    print("Etykiety", list(some_labels))
    # prognozy
    print("Prognozy model maszyny wektorów nośnych:", grid_search_svr.predict(some_data_prepared))
    # pipeline z przekształcaniem danych i tworzeniem prognoz modelu svc
    custom_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('scaler', StandardScaler()),
        ('svr', SVR())])
    # get test and train data from fetch_data
    svr_reg_2 = custom_pipeline.fit(train_set, train_labels)
    scores_svr_2 = cross_val_score(svr_reg_2, train_tr, train_labels,
                                   scoring="neg_mean_squared_error", cv=10)
    data.display_scores(scores_svr_2, "Wyniki dla modelu maszyny wektorów nośnych z samego pipeline:")
    data.saving_scores_to_file(scores_svr_2,
                               custom_start_print="Wyniki dla modelu maszyny wektorów nośnych metoda 2:\n")
    # etykiety
    print("Etykiety", list(some_labels))
    # prognozy
    print("Prognozy model maszyny wektorów nośnych z samego pipeline:", custom_pipeline.predict(some_data))
    # sprawdzenie modelu za pomocą zbioru danych testowych svr 1
    X_test_prepared = num_pipeline.transform(test_set)
    final_pred_svr_1 = svr_reg_1.predict(X_test_prepared)

    confidence = 0.95
    squared_errors = (final_pred_svr_1 - Y_test) ** 2
    final_rmse_svr_1 = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, loc=squared_errors.mean(),
                                                scale=stats.sem(squared_errors)))
    print(f"Finalny błąd rmse svr 1: {final_rmse_svr_1}")
    # sprawdzenie modelu za pomocą zbioru danych testowych svr 1
    X_test_prepared = num_pipeline.transform(test_set)
    final_pred_svr_2 = svr_reg_2.predict(X_test_prepared)

    confidence = 0.95
    squared_errors = (final_pred_svr_2 - Y_test) ** 2
    final_rmse_svr_2 = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1, loc=squared_errors.mean(),
                                                scale=stats.sem(squared_errors)))
    print(f"Finalny błąd rmse svr 2: {final_rmse_svr_2}")
    # save searching model to joblib path
    joblib.dump(final_rmse_svr_1, r"../../Output/mój_model.pkl")
