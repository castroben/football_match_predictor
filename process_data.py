from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import ElasticNet, Ridge, RidgeClassifier
import pandas as pd
import numpy as np

def classify(dataset):
    category = dataset.split('.')[0]
    f = open('results/' + category + '.txt', 'w')

    df = pd.read_csv('classification_data/classification_' + dataset)
    X = df.drop('Result', axis=1)
    y = df['Result']
    
    # dividing X, y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

    # defining parameter range
    f.write('Random Forest\n')
    param_grid = {
                'n_estimators': [10, 50, 100, 200],  
                'criterion': ['gini', 'entropy'], 
                'max_features':['sqrt', 'log2'],
                'bootstrap': [True, False]
            }  
    
    grid = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, cv=3) 
    
    # fitting the model for grid search 
    grid.fit(X_train, y_train) 
    
    # print best parameter after tuning 
    f.write(str(grid.best_params_))
    grid_predictions = grid.predict(X_test) 
    
    # print classification report 
    f.write(str(classification_report(y_test, grid_predictions)))

    f.write('\n Ridge')
    param_grid = {
        'alpha' : [1.0, 2.0, 5.0, 10.0],
        'solver': ['auto', 'svd', 'saga'],
        'random_state': [0, 5, 10]
    }
    grid = GridSearchCV(RidgeClassifier(), param_grid=param_grid, cv=3)
    grid.fit(X_train, y_train)
    f.write(str(grid.best_params_))
    grid_predictions = grid.predict(X_test)
    f.write(str(classification_report(y_test, grid_predictions)))
    f.close

def regress(dataset):
    category = dataset.split('.')[0]
    f = open('regression_results/' + category + '.txt', 'w')

    df = pd.read_csv('regression_data/regression_' + dataset)
    X = df.drop('Goals', axis=1)
    y = df['Goals']

    # dividing X, y into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)

    # RandomForestRegressor
    
    param_grid = {
        'alpha' : [1.0, 2.0, 5.0, 10.0],
        'solver': ['auto', 'svd', 'saga'],
        'random_state': [0, 5, 10]
    }
    grid = GridSearchCV(Ridge(), param_grid=param_grid, cv=3)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    predictions = [round(num) for num in model.predict(X_test)]
    print(predictions)
    print(print(list(y_test)))
    print(model.get_params())
    print(grid.best_params_)
    print(mean_absolute_error(y_test, predictions))

    f.close

if __name__ == '__main__':
    # datasets = ['england.csv', 'spain.csv', 'germany.csv', 'italy.csv', 'france.csv', 'world.csv', 'all.csv']
    # for dataset in datasets:
    #     classify(dataset)
    #classify('all.csv')
    regress('all.csv')