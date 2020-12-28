import numpy as np
from matplotlib import pyplot
from sklearn import svm, metrics
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate

from Preprocess import *
from mlxtend.evaluate import paired_ttest_kfold_cv

NUMBER_FOLD = 10


class CompareAlgorithm:

    def __init__(self):
        self.pre = PreProcessing()
        self.scoring = ['precision', 'recall', 'f1', 'accuracy']
        # Perform 10 fold validation
        self.kfold = KFold(n_splits=NUMBER_FOLD, random_state=100)
        self.thresholds = [0.3, 0.5, 0.7, 0.9, 0.99]

    def main(self):
        self.pre_process()
        self.examine_svm_kernels()
        self.rf_examine_criterion()
        self.rf_examine_n_estimators()
        print("============================================================")
        print("============================================================")
        print("SVM vs. Random Forest:")
        print()

        self.best_svm()
        self.best_rf()
        self.statistical_significance_tests()
        self.sensitivity_analysis(self.svm_model)
        self.sensitivity_analysis(self.rf_model)

    def pre_process(self):
        self.pre.pre_process()

    def examine_svm_kernels(self):
        print("==================== SVM - Examine Parameters ====================")
        print()
        print(" 1. ---------- Examine SVM Kernels: ----------")
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        for i in range(len(kernels)):
            svm_model = svm.SVC(kernel=kernels[i], probability=True)
            scores_kfold_svm = cross_validate(estimator=svm_model, X=self.pre.x, y=self.pre.y, cv=self.kfold,
                                              scoring=self.scoring, return_train_score=False)
            CompareAlgorithm.evaluate_metric(scores_kfold_svm, "{} kernel SVM".format(kernels[i]))
        print()
        print("SVM's best operators:  ")
        print("Kernel: rbf ")

    def rf_examine_criterion(self):
        print("==================== RF - Examine Parameters ====================")
        print()
        print(" 1. ---------- Examine criterion: ----------")
        criterion = ['gini', 'entropy']
        for i in range(len(criterion)):
            RF_model = RandomForestClassifier(n_estimators=100, criterion=criterion[i])
            scores_kfold_RF = cross_validate(estimator=RF_model, X=self.pre.x, y=self.pre.y, cv=self.kfold,
                                             scoring=self.scoring, return_train_score=False)
            CompareAlgorithm.evaluate_metric(scores_kfold_RF, "{} criterion RF".format(criterion[i]))

    def rf_examine_n_estimators(self):
        print("2. ---------- Examine n_estimators: ----------")
        n_estimators = [50, 100, 150]
        for i in range(len(n_estimators)):
            RF_model = RandomForestClassifier(n_estimators=n_estimators[i], criterion='entropy', random_state=42)
            scores_kfold_RF = cross_validate(estimator=RF_model, X=self.pre.x, y=self.pre.y, cv=self.kfold,
                                             scoring=self.scoring, return_train_score=False)
            CompareAlgorithm.evaluate_metric(scores_kfold_RF, "{} n_estimators RF".format(n_estimators[i]))
        print()
        print("RF's best operators:  ")
        print("criterion: entropy ")
        print("n_estimators: 100 ")

    @staticmethod
    def evaluate_metric(model, model_name):
        print("*** {} metrics *** ".format(model_name))
        print("Accuracy: %0.2f " % (np.mean(model['test_accuracy']) * 100))
        print("Avg precision: {}".format(np.mean(model['test_precision'])))
        print("Avg recall: {}".format(np.mean(model['test_recall'])))
        print()

    def best_svm(self):
        self.svm_model = svm.SVC(kernel='rbf', probability=True)
        scores_kfold_svm = cross_validate(estimator=self.svm_model, X=self.pre.x, y=self.pre.y, cv=self.kfold,
                                          scoring=self.scoring, return_train_score=False)
        CompareAlgorithm.evaluate_metric(scores_kfold_svm, "SVM")

    def best_rf(self):
        self.rf_model = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=42)
        scores_kfold_RF = cross_validate(estimator=self.rf_model, X=self.pre.x, y=self.pre.y, cv=self.kfold,
                                         scoring=self.scoring, return_train_score=False)
        CompareAlgorithm.evaluate_metric(scores_kfold_RF, "RF")

    def sensitivity_analysis(self, model):
        print("============================================================")
        print("============================================================")
        print("Sensitivity Analysis:")
        print()
        predict_prob = cross_val_predict(estimator=model, X=self.pre.x, y=self.pre.y, cv=self.kfold,
                                         method='predict_proba')
        list_accuracy, list_precision, list_recall = [], [], []
        predict_prob = predict_prob[:, 1]
        for i in self.thresholds:
            list_prob = []
            for idx, prob in enumerate(predict_prob):
                if predict_prob[idx] < i:
                    list_prob.append(0)
                else:
                    list_prob.append(1)
            acc = metrics.accuracy_score(self.pre.y, list_prob)
            list_accuracy.append(acc)
            precision = metrics.precision_score(self.pre.y, list_prob)
            list_precision.append(precision)
            recall = metrics.recall_score(self.pre.y, list_prob)
            list_recall.append(recall)
            print(f"Thresholds = {i}:")
            print("Accuracy: %0.2f " % (acc))
            print("Precision: {}".format(precision))
            print("Recall: {}".format(recall))
            print()
        self.plot_threshold(list_accuracy, list_precision, list_recall)

    def plot_threshold(self, list_accuracy, list_precision, list_recall):
        # plot loss
        pyplot.subplot(3, 1, 1)
        pyplot.scatter(self.thresholds, list_accuracy, color= "green")
        pyplot.xlabel('Thresholds')
        pyplot.ylabel('Accuracy')
        # plot discriminator accuracy
        pyplot.subplot(3, 1, 2)
        pyplot.scatter(self.thresholds, list_precision)
        pyplot.xlabel('Thresholds')
        pyplot.ylabel('Precision')
        pyplot.subplot(3, 1, 3)
        pyplot.scatter(self.thresholds, list_recall, color= "red")
        pyplot.xlabel('Thresholds')
        pyplot.ylabel('Recall')
        # save plot to file
        pyplot.savefig('plot_threshold.png')
        pyplot.show()
        pyplot.close()

    def statistical_significance_tests(self):
        print("============================================================")
        print("============================================================")
        print("K-fold cross-validated paired t-test:")
        print()

        t, p = paired_ttest_kfold_cv(estimator1=self.svm_model,
                                     estimator2=self.rf_model,
                                     X=self.pre.x, y=self.pre.y,
                                     random_seed=42)
        print('t statistic: %.3f' % t)
        print('p value: %.3f' % p)


if __name__ == '__main__':
    compare_algorithm = CompareAlgorithm()
    compare_algorithm.main()
