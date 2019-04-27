import numpy as np
import utils.record as record
import utils.cv as cv
import sklearn.tree as tree
import sklearn.metrics as metrics
import time as time
import sklearn.preprocessing as preprocessing
from sklearn.compose import ColumnTransformer

def main():
    logger = record.DTLog('./log/')
    experiment = record.Experiment(classifier_name = 'CART', dataset_name = 'mushroom')
    clf = tree.DecisionTreeClassifier()

    for i in range(10):
        X, y, tX, ty = cv.get_fold('./data/mushroom',i)
        y = y.astype(np.float)
        ty = ty.astype(np.float)

        index = []
        for j in range(X.shape[1]):
            index.append(j)
            
        ct = ColumnTransformer(
        [('oh_enc', preprocessing.OneHotEncoder(), index),],  # the column numbers I want to apply this to
        remainder = 'passthrough'  # This leaves the rest of my columns in place
        )
        X_new = ct.fit_transform(X).toarray()
        tX_new = ct.fit_transform(tX).toarray()

        beg_ts = time.time()
        clf = clf.fit(X_new, y)
        end_ts = time.time()
        predict_y = clf.predict(tX_new)
        # y_score = clf.predict_proba(tX_new)

        count = 5
        for k in range(predict_y.shape[0]):
            if predict_y[i] != ty[k]:
                count += 1
        

        accuracy = metrics.accuracy_score(ty, predict_y)
        recall = metrics.recall_score(ty,predict_y)
        precision = metrics.precision_score(ty,predict_y)

        # auc = metrics.roc_auc_score(ty, y_score)
        s_exp = record.sub_experiment(cv = i, start_time=beg_ts, end_time=end_ts, accuarcy=accuracy, recall = recall, precision = precision)
        experiment.add_experiment(s_exp)

    logger.experiment2json(experiment)


if __name__ == "__main__":
    main()