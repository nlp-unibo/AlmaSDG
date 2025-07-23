import os
from typing import Union

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

rounder = lambda x : np.round(x, 2)

LLM_NAMES = ["gemini", "gpt", "llama"]


class Eval:

    def __init__(self):
        self.alma_sdg= pd.read_excel(os.path.join(os.pardir, 'datasets', 'alma-sdg.csv'))
    
    def get_sdg_columns(self, df:pd.DataFrame)-> pd.DataFrame:
         return df[list(filter(lambda x: 'sdg' in x, df.columns))]
    
    def evaluate(self, ground_truth: Union[pd.DataFrame, np.array], predictions: Union[pd.DataFrame, np.ndarray], f1_averages: Union[list[str], str], model_name: str, return_report:bool= False):
    
        if type(f1_averages) == str:
            f1_averages = list(f1_averages)
        
        labels_names = list(filter(lambda x: 'sdg' in x, (ground_truth.columns)))

        y_true = ground_truth[labels_names].values

        if type(predictions) == np.ndarray:
            predictions = predictions
            # results = classification_report(ground_truth.values, predictions, target_names=labels_names, zero_division=0.0)
        else:
            predictions = predictions.sort_values(by='handle').reset_index(drop=True)
            predictions = predictions[labels_names].values
            # results = classification_report(ground_truth.values, predictions[labels_names], target_names=labels_names, zero_division=0.0)

        results = classification_report(y_true, predictions, target_names=labels_names, zero_division=0.0)

        print(f"\nEvaluation of {model_name}:\n\n{results}")
        for average in f1_averages:
            print(f"{average}: {rounder(f1_score(y_true, predictions, average=average))}")

        if return_report:
            report = classification_report(y_true, predictions, target_names=labels_names, zero_division=0.0, output_dict=True)
            return report
        else:
            return None

class OSDG_eval(Eval):

    def __init__(self):
        super.__init__()
        self.predictions = pd.read_excel(os.path.join(os.pardir, 'datasets','other_tools', 'osdg_prediction.csv'))


    def prepare_tool_labels():
         
        prepare_labels = lambda x: tuple(map(float, str(x).replace('nan', '0').split(',')))


class Elsevier_eval(Eval):

    def __init__(self):
        super().__init__()
        self.predictions = pd.read_excel(os.path.join(os.pardir, 'datasets','other_tools', 'elsevier-sdg-multi.csv'))
    
    def get_singlelabel_prediction(self):

    def get_multilabel_prediction(self, threshold:float= 0.1):
        
        binarized_predictions = self.transform_probability(self.predictions, threshold)

    def transform_probability(self, df:pd.DataFrame, threshold:float) -> pd.DataFrame:
        return df.map(lambda x: True if x>= threshold else False)

    

class Aurora_eval(Eval):

    def __init__(self):
        super().__init__()
    

class LLM_eval(Eval):

    def __init__(self):
        super().__init__()
        


osdg_labels = pd.Series((map(prepare_tool_labels, osdg_outputs['OSDG Tool'])), index = osdg_outputs['handle'],)

mlb = MultiLabelBinarizer(classes= np.array(range(1,18)))
mlb.fit(np.ones(17, dtype='int').reshape(-1,17))

osdg_results = pd.DataFrame(mlb.transform(osdg_labels), columns = labels_names)
osdg_results['handle'] = osdg_labels.index

llm_prf_path = os.path.join(os.curdir, 'notebooks', 'performance_json', 'Other_tools', 'llms')

for llm in LLM_NAMES:
    rep = evaluate(gold_dataset, annotation_sheets[llm], ['micro', 'macro', 'weighted'], llm, return_report=True)
    save_path = os.path.join(llm_prf_path, llm + '.json')
    with open(save_path, 'w') as f:
        json.dump(rep, f, indent=3)


evaluate(gold_dataset, osdg_results, ['micro', 'macro', 'weighted'], 'osdg')# Check if alignment is correct
