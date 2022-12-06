# modules for this classification program
import pandas as pd
import warnings
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, accuracy_score
import pickle

# suppress warnings
warnings.filterwarnings('ignore')

# constant variable for model reproducibility
RANDOM_SEED = 2

# import the .csv files which is being created from homework.ipynb file
train_df = pd.read_csv('train_df.csv', index_col=0) # train data & unseen data
test_df = pd.read_csv('test_df.csv', index_col=0) # simulate future unseen data

# prepare X (features) and y (label) for model training
X = train_df.iloc[:, :5]
y = train_df['Severity']

# split the data set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=RANDOM_SEED)

# write a function for model evaluation
def model_classification_report(y_test, model_y_pred):
    '''Prints out confusion matrix and classification report of the trained model.
    
    Parameters
    ----------
    y_test : pandas.core.series.Series
        The actual predicted values from train_df (a.k.a ground truth)
    model_y_pred : numpy.ndarray
        A 1D numpy array of model prediction values

    '''
    y_test=y_test
    model_confusion = metrics.confusion_matrix(y_test, model_y_pred, labels=(1,0))
    model_classification_report = classification_report(y_test, model_y_pred)
    print(f'Confusion Matrix: \n{model_confusion}\n')
    print(f'Classification Report: \n{model_classification_report}\n')

# building model 1: logistic regression classifier
logreg_steps=[('std_scaler',StandardScaler()),
                ('logreg', LogisticRegression(random_state=RANDOM_SEED))]

p1 = Pipeline(logreg_steps)

logreg_params = {
    'logreg__penalty':['l1', 'l2'],
    'logreg__C':[0.01, 0.1, 1.0],
    'logreg__solver':['lbfgs', 'liblinear'],
    'logreg__max_iter':list(range(50,200,10))
}

cv = StratifiedKFold(n_splits=10, random_state=RANDOM_SEED, shuffle=True)

logreg_gridsearch = GridSearchCV(p1, param_grid=logreg_params, n_jobs=-1, cv=cv, verbose=0)

logreg_gridsearch.fit(X_train, y_train)
logreg_y_pred = logreg_gridsearch.predict(X_test)
logreg_auc = metrics.roc_auc_score(y_test, logreg_y_pred)

print(logreg_gridsearch.best_params_)
print(f'logreg accuracy: {logreg_gridsearch.best_score_}')
print(f'logreg AUC: {logreg_auc}')

# evaluate model 1: logistic regression classifer
model_classification_report(y_test, logreg_y_pred)

# builing model 2: random forest classifier
rf_clf = RandomForestClassifier(random_state=RANDOM_SEED)

rf_clf_params = {
    'criterion':['gini', 'entropy', 'log_loss'],
    'max_depth':list(range(5,10)),
    'min_samples_split':list(range(2,5)),
    'min_samples_leaf':list(range(2,5))
}

cv = StratifiedKFold(n_splits=10, random_state=RANDOM_SEED, shuffle=True)

rf_clf_gridsearch = GridSearchCV(rf_clf, param_grid=rf_clf_params, n_jobs=-1, cv=cv, verbose=0)

rf_clf_gridsearch.fit(X_train, y_train)
rf_clf_y_pred = rf_clf_gridsearch.predict(X_test)
rf_clf_auc = metrics.roc_auc_score(y_test, rf_clf_y_pred)

print(rf_clf_gridsearch.best_params_)
print(f'rf_clf accuracy: {rf_clf_gridsearch.best_score_}')
print(f'rf_clf AUC: {rf_clf_auc}')

# evaluate model 2: random forest classifier
model_classification_report(y_test, rf_clf_y_pred)

# building model 3: support vector machine classifier
svm_clf_steps=[('std_scaler',StandardScaler()),
                ('svm_clf', SVC(random_state=RANDOM_SEED))]

p2 = Pipeline(svm_clf_steps)

svm_clf_params = {
    'svm_clf__C':[0.1, 1.0, 10, 100],
    'svm_clf__kernel':['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
    'svm_clf__gamma':['scale', 'auto']
}

cv = StratifiedKFold(n_splits=10, random_state=RANDOM_SEED, shuffle=True)

svm_clf_gridsearch = GridSearchCV(p2, param_grid=svm_clf_params, n_jobs=-1, cv=cv, verbose=0)

svm_clf_gridsearch.fit(X_train, y_train)
svm_clf_y_pred = svm_clf_gridsearch.predict(X_test)
svm_clf_auc = metrics.roc_auc_score(y_test, svm_clf_y_pred)

print(svm_clf_gridsearch.best_params_)
print(f'logreg accuracy: {svm_clf_gridsearch.best_score_}')
print(f'logreg AUC: {svm_clf_auc}')

# evalute model 3: catboost classifier
model_classification_report(y_test, svm_clf_y_pred)

# iterations performed
# 1. choosing 4 features (all features except BI-RADS) did not improve AUC
# 2. choosing 4 features and increasing test size to 0.3, decreases the AUC
# 3. choosing 4 features and decreasing test size to 0.15, did not improve AUC but produces higher AUC than iteration 2
# 4. Going back to 5 features and decreasing test size to 0.15, improves AUC by around 0.002

# choosing the best model
# choose the training model which produces the lowest FP and FN
# in this case, random forest classifier is the best model

# rebuild & evaluate best model on test_df
best_model = RandomForestClassifier(random_state=RANDOM_SEED, criterion='gini',
                                    max_depth=9, min_samples_leaf=4, min_samples_split=2)

unseen_x = test_df.iloc[:, :5]
unseen_y = test_df['Severity']

best_model.fit(X, y)
best_model_y_pred = best_model.predict(unseen_x)
best_model_score = accuracy_score(unseen_y, best_model_y_pred)
best_model_auc = metrics.roc_auc_score(unseen_y, best_model_y_pred)

print(f'best_model accuracy: {best_model_score}')
print(f'best_model AUC: {best_model_auc}')

best_model_confusion = metrics.confusion_matrix(unseen_y, best_model_y_pred, labels=(1,0))
best_model_classification_report = classification_report(unseen_y, best_model_y_pred)

print(f'Confusion Matrix: \n{best_model_confusion}\n')
print(f'Classification Report: \n{best_model_classification_report}\n')

# pickling the best model file for deployment
pickle.dump(best_model, open('best_model.pkl', 'wb'))
pickled_model = pickle.load(open('best_model.pkl', 'rb'))
pickled_model.predict(unseen_x)