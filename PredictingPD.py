import numpy as np
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from Preprocess import *
from mlxtend.evaluate import paired_ttest_kfold_cv


# region Evaluation methods
def evaluate_metric(model, model_name):
    print("*** {} metrics *** ".format(model_name))
    print("Accuracy: %0.2f " % (np.mean(model['test_accuracy']) * 100))
    print("Avg precision: {}".format(np.mean(model['test_precision'])))
    print("Avg recall: {}".format(np.mean(model['test_recall'])))
    print()


def examine_svm_kernels():
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    for i in range(len(kernels)):
        svm_model = svm.SVC(kernel=kernels[i], probability=True)
        scores_kfold_svm = cross_validate(estimator=svm_model, X=x, y=y, cv=kfold, scoring=scoring,
                                          return_train_score=False)
        evaluate_metric(scores_kfold_svm, "{} kernel SVM".format(kernels[i]))


def rf_examine_criterion():
    criterion = ['gini', 'entropy']
    for i in range(len(criterion)):
        RF_model = RandomForestClassifier(n_estimators=100, criterion=criterion[i])
        scores_kfold_RF = cross_validate(estimator=RF_model, X=x, y=y, cv=kfold, scoring=scoring,
                                         return_train_score=False)
        evaluate_metric(scores_kfold_RF, "{} criterion RF".format(criterion[i]))


def rf_examine_n_estimators():
    n_estimators = [50, 100, 150]
    for i in range(len(n_estimators)):
        RF_model = RandomForestClassifier(n_estimators=n_estimators[i], criterion='entropy')
        scores_kfold_RF = cross_validate(estimator=RF_model, X=x, y=y, cv=kfold, scoring=scoring,
                                         return_train_score=False)
        evaluate_metric(scores_kfold_RF, "{} n_estimators RF".format(n_estimators[i]))
# endregion


scoring = ['precision', 'recall', 'f1', 'accuracy']

# Perform 10 fold validation
kfold = KFold(n_splits=10, random_state=100)

# Examine SVM Kernels
print("==================== SVM - Examine Parameters ====================")
print()
print(" 1. ---------- Examine SVM Kernels: ----------")
examine_svm_kernels()
print()
print("SVM's best operators:  ")
print("Kernel: rbf ")

# Examine RF parameters

print("==================== RF - Examine Parameters ====================")
print()
print(" 1. ---------- Examine criterion: ----------")
rf_examine_criterion()
print("2. ---------- Examine n_estimators: ----------")
rf_examine_n_estimators()

print()
print("RF's best operators:  ")
print("criterion: entropy ")
print("n_estimators: 100 ")

print("============================================================")
print("============================================================")
print("SVM vs. Random Forest:")
print()
svm_model = svm.SVC(kernel='rbf', probability=True)
scores_kfold_svm = cross_validate(estimator=svm_model, X=x, y=y, cv=kfold, scoring=scoring,
                                  return_train_score=False)
evaluate_metric(scores_kfold_svm, "SVM")

RF_model = RandomForestClassifier(n_estimators=100, criterion='entropy')
scores_kfold_RF = cross_validate(estimator=RF_model, X=x, y=y, cv=kfold, scoring=scoring,
                                 return_train_score=False)
evaluate_metric(scores_kfold_RF, "RF")

print("============================================================")
print("============================================================")
print("K-fold cross-validated paired t-test:")
print()

svm_model.fit(x_train, y_train)
RF_model.fit(x_train, y_train)
t, p = paired_ttest_kfold_cv(estimator1=svm_model,
                             estimator2=RF_model,
                             X=x, y=y,
                             random_seed=1)
print('t statistic: %.3f' % t)
print('p value: %.3f' % p)
