import json 
import datetime as datetime
import os 
import time as time

class DTLog(object):

    def __init__(self, target_path = None):
        self.filepath = self._check_path(target_path)

    # check whether input file path is None and 
    # contains forward slash at tail.
    def _check_path(self, target_path):
        if target_path == None:
            return './log/'
        else:
            return os.path.join(target_path,'')


    # generate filename 
    # in format {calssifier name}-{dataset name}_{current time}
    def _generate_filename(self, classifier_name, dataset_name):
        formter = "{}-{}_{:%Y-%m-%d_%H-%M-%S}.json"
        return formter.format(classifier_name, \
            dataset_name, datetime.datetime.now())
        
    # convert expertiment object to json and save to taget path.
    def experiment2json(self,experiment):
        filename = self._generate_filename(experiment.classifier_name, \
             experiment.dataset_name)

        with open(os.path.join(self.filepath,filename), 'w') as outfile:
            json.dump(experiment, outfile,default=lambda o:o.__dict__)




class Experiment(object):
    '''
    Json format:
    {
        classifier_name: xxx
        dataset_name: xxx
        experiments: [
            {
                cv: 1
                start_time: 000000,
                end_time: 000000,
                run_time: 000000,
                recall: 00000,
                precision: 00000,
                accuarcy: 00000,
                auc: 000000
            },
        ]
    }
    '''

    def __init__(self, classifier_name=None, dataset_name=None):

        if classifier_name == None or not isinstance(classifier_name,str):
            self.classifier_name = "unknown"
        else:
            self.classifier_name = classifier_name

        if dataset_name == None or not isinstance(classifier_name,str):
            self.dataset_name = "unknown"
        else:
            self.dataset_name = dataset_name

        self.experiments = []

    def add_experiment(self, experiment):
        self.experiments.append(experiment)
    


class sub_experiment(object):
    def __init__(self, cv, start_time, end_time, recall = None, precision = None,
                 accuarcy = None, auc = None, run_time=None):
        self.cv = cv
        self.start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        self.end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
        self.runtime = end_time - start_time
        self.recall = recall
        self.precision = precision
        self.accuarcy = accuarcy
        # self.auc = auc