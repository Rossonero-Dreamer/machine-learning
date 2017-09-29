import pandas as pd
import numpy as np
from time import time

#training and prediction pipeline
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    results = {}

    #fit the model with training data of sample size
    start = time()
    learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time()
    results['train_time'] = end - start

    #obtain prediction on test data and first 300 training data
    start = time()
    pred_test = learner.predict(X_test)
    pred_train = learner.predict(X_train[:300])
    end = time()
    results['pred_time'] = end - start

    #evaluate scores
    results['acc_train'] = accuracy_score(y_train[:300], pred_train)
    results['acc_test'] = accuracy_score(y_test, pred_test)
    results['f_train'] = fbeta_score(y_train[:300], pred_train, 0.5)
    results['f_test'] = fbeta_score(y_test, pred_test, 0.5)

    #output
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))

    return results

#load the data
data = pd.read_csv('census.csv')

#below are the data fields
#age: continuous
#workclass: discrete
#education: discrete
#education-num: continuous
#marital-status: discrete
#occupation: discrete
#relationship: discrete
#race: discrete
#sex: discrete
#capital-gain: continuous
#capital-loss: continuous
#hours-per-week: continuous
#native-country: discrete

#separate features and targets
income_raw = data.income
features_raw = data.drop('income', axis=1)

#apply log transform to skewed features
skewed = ['capital-gain','capital-loss']
feature_log_transformed = pd.DataFrame(data = features_raw)
feature_log_transformed[skewed] = feature_log_transformed[skewed].apply(lambda x:np.log(x+1))

#normalize the data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
numericals = ['age','education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
feature_log_minmax = pd.DataFrame(data = feature_log_transformed)
feature_log_minmax[numericals] = scaler.fit_transform(feature_log_minmax[numericals])

#encode categorical values
income = (income_raw == '>50K').astype(int)
features = pd.get_dummies(feature_log_minmax)

#split the training and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2
                                                    , random_state=22)

#evaluate three learners
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

clf_a = GaussianNB()
clf_b = AdaBoostClassifier()
clf_c = SVC()

sample_100 = len(y_train)
sample_10 = sample_100/10
sample_1 = sample_100/100

#collect results from training
#results = {}
#for clf in [clf_a, clf_b, clf_c]:
#    clf_name = clf.__class__.__name__
#    results[clf_name]={}
#    for i, samples in enumerate([sample_1, sample_10]):
#        results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_test, y_test)

#print results
#for clf_name in results:
#    print(results[clf_name])

#use gridsearch to finetune the chosen model
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer

clf = AdaBoostClassifier(random_state=22)
parameters = {'learning_rate':[0.1,1,3,5,10,20]}
scorer = make_scorer(fbeta_score, beta=0.5)
grid_obj = GridSearchCV(clf, parameters, scoring = scorer)
grid_obj.fit(X_train, y_train)
best_clf = grid_obj.best_estimator_

#compare best to normal
pred = (clf.fit(X_train, y_train)).predict(X_test)
fscore = fbeta_score(y_test, pred, beta=0.5)
pred_best = best_clf.predict(X_test)
fscore_best = fbeta_score(y_test, pred_best, beta=0.5)
print(fscore, fscore_best)

#display important features
#print(best_clf.feature_importances_)
