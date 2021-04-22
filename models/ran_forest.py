import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, make_scorer, confusion_matrix
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import OneClassSVM


datos = pd.read_csv('data/banksim_adj.csv')

feature_columns = ['age', 'amount', 'M', 'es_barsandrestaurants', 'es_contents', 'es_fashion',
                    'es_food', 'es_health', 'es_home', 'es_hotelservices', 'es_hyper',
                    'es_leisure', 'es_otherservices', 'es_sportsandtoys', 'es_tech',
                    'es_transportation', 'es_travel']
target_column = ['fraud']
X = datos.drop("fraud", inplace = False, axis = 1).values
y = datos[["fraud"]].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 10, stratify = y)

models = ["Isolation Forest", "One Class SVM","Local Outlier Factor","K-means Outlier Detector"]
precisions = []
recalls = []
f1_scores = []

def outlier_coding_to_labels(y_outliers):
    y_labels = y_outliers.copy()
    y_labels[y_labels == 1] = 0
    y_labels[y_labels == -1] = 1
    return y_labels
    
def recall_w_precision_threshold(y, y_pred, prec_threshold = 0.3, outlier_case = False):
    if outlier_case:
        y_pred = outlier_coding_to_labels(y_pred)
    precision = precision_score(y, y_pred)
    if precision >= prec_threshold:
        return recall_score(y, y_pred)
    else:
        return 0

my_scorer = make_scorer(recall_w_precision_threshold, prec_threshold = 0.2, outlier_case = True)
parameters = {'iforest__n_estimators':[10,20,50,100],
                'iforest__contamination':[0.02, 0.05, 0.1]}
pipe_rf = Pipeline(steps = [
    ("iforest",IsolationForest())
    ])
gs = GridSearchCV(estimator = pipe_rf, n_jobs = 2, 
                    param_grid = parameters,
                    scoring = my_scorer,
                    cv = StratifiedKFold(n_splits = 6, shuffle = True, random_state = 239)
                )

gs.fit(X_train, y_train)
best_model: IsolationForest = gs.best_estimator_

best_model.fit(X_train)
y_test_pred = outlier_coding_to_labels(best_model.predict(X_test))


precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)
precisions.append(precision)
recalls.append(recall)
f1_scores.append(f1)
pickle.dump(best_model, open('models/RFregression.pkl', 'wb'))




#my_scorer = make_scorer(recall_w_precision_threshold, prec_threshold = 0.2, outlier_case = True)
#parameters = [{
#                'svm__kernel':['rbf','sigmoid'], 
#                'svm__gamma':['scale','auto',0.1,0.5]
#                },
#                {
#                'svm__kernel':['poly'],
#                'svm__gamma':['scale','auto',0.1,0.5],
#                'svm__degree':[2,3,5]
#                }]

#pipe = Pipeline(steps = [("scaler",MinMaxScaler()), ("svm",OneClassSVM())
#    ])
#gs = GridSearchCV(estimator = pipe, n_jobs = 2, 
#                    param_grid = parameters,
#                    scoring = my_scorer,
#                    cv = StratifiedKFold(n_splits = 6, shuffle = True, random_state = 239)
#               )

#gs.fit(X_train, y_train)
#best_model: OneClassSVM = gs.best_estimator_
#print("El mejor score fue {} y se obtuvo con los siguientes parámetros:\n {}".format(gs.best_score_, gs.best_params_))
#best_model.fit(X_train)
#y_test_pred = outlier_coding_to_labels(best_model.predict(X_test))


#precision = precision_score(y_test, y_test_pred)
#recall = recall_score(y_test, y_test_pred)
#f1 = f1_score(y_test, y_test_pred)
#precisions.append(precision)
#recalls.append(recall)
#f1_scores.append(f1)

#print("Sobre el conjunto de prueba, para el modelo One Class SVM se obtuvieron las siguientes métricas:")
#print("Precision: {}".format(precision))
#print("Recall: {}".format(recall))
#print("F1: {}".format(f1))

#print("\n Matriz de confusión:")
#confusion_matrix(y_test, y_test_pred, labels = [0,1])
#feature_columns = ["mnth", "new_time", "season", "holiday", "weekday", "workingday", "weathersit", "temp", "atemp", "hum", "windspeed"]
#target_column = "cnt"
#y_new = datos[target_column]
#X_new = datos[feature_columns]

#ind_new = datos["yr"] == 0
#X_train_n, y_train_n = X_new[ind_new], y_new[ind_new]
#X_test_n, y_test_n = X_new[~ind_new], y_new[~ind_new]

#assert X_new.shape[0] == X_train_n.shape[0] + X_test_n.shape[0]

#col = ['season','new_time','workingday',"weathersit", 'temp','atemp','hum']
#x_trainf = X_train_n[col]
#x_testf = X_test_n[col]

#pipe_rf = Pipeline(steps=[("scaler", MinMaxScaler()),
#    ("rfmodel", RandomForestRegressor(n_estimators=4, max_depth=10))
#])

#pipe_rf.fit(X_train, y_train)

#pickle.dump(best_model, open('models/RFregression.pkl', 'wb'))