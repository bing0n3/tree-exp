import numpy as pd 
import utils.record as record
import utils.cv as cv
import sklearn.tree as tree
import sklearn.metrics as metrics
import time as time

def main():
    logger = record.DTLog('./log/')
    experiment = record.Experiment(classifier_name = 'CART', dataset_name = 'iris')
    clf = tree.DecisionTreeClassifier()

    for i in range(10):
        X, y, tX, ty = cv.get_fold('./data/iris',i)
        clf = tree.DecisionTreeClassifier()
        beg_ts = time.time()
        clf = clf.fit(X, y)
        end_ts = time.time()
        predict_y = clf.predict(tX)
        y_score = clf.predict_proba(tX)
        accuracy = metrics.accuracy_score(ty, predict_y)
        recall = metrics.recall_score(ty,predict_y)
        precision = metrics.precision_score(ty,predict_y)
        # auc = metrics.roc_auc_score(ty, y_score)
        s_exp = record.sub_experiment(cv = i,start_time=beg_ts,end_time=end_ts, accuarcy=accuracy, recall = recall, precision = precision)
        experiment.add_experiment(s_exp)

    logger.experiment2json(experiment)


if __name__ == "__main__":
    main()