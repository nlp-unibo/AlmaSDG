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
        self.alma_sdg= pd.read_csv(os.path.join(os.pardir, 'dataset', 'alma_sdg.csv')).sort_values(by='handle').reset_index(drop=True)
    
    def get_sdg_columns(self, df:pd.DataFrame)-> pd.DataFrame:
         return df[list(filter(lambda x: 'sdg' in x, df.columns))]
    
    def evaluate(self, predictions: Union[pd.DataFrame, np.ndarray], f1_averages: Union[list[str], str], model_name: str):
        
        ground_truth = self.alma_sdg

        if type(f1_averages) == str:
            f1_averages = list(f1_averages)
        
        labels_names = list(filter(lambda x: 'sdg' in x, (ground_truth.columns)))

        y_true = ground_truth[labels_names].values

        if type(predictions) == np.ndarray:
            predictions = predictions
        else:
            # predictions = predictions.sort_values(by='handle').reset_index(drop=True)
            predictions = predictions[labels_names].values

        results = classification_report(y_true, predictions, target_names=labels_names, zero_division=0.0)

        print(f"\nEvaluation of {model_name}:\n\n{results}")
        for average in f1_averages:
            print(f"{average}: {rounder(f1_score(y_true, predictions, average=average))}")

class OSDG_eval(Eval):

    def __init__(self):
        super().__init__()
        self.predictions = pd.read_csv(os.path.join(os.pardir, 'validation_results', 'osdg_prediction.csv')).sort_values(by='handle').reset_index(drop=True)
        self.mlb = MultiLabelBinarizer(classes= np.array(range(1,18)))
        self.model_name= 'OSDG'
        
        self.mlb.fit(np.ones(17, dtype='int').reshape(-1,17))

    def prepare_tool_labels(self):
        
        labels = self.predictions['OSDG Tool']
        prepare_labels = lambda x: list(map(float, str(x).replace('nan', '0').split(',')))
        labels = labels.apply(prepare_labels)

        y_hat = self.mlb.transform(labels)

        return y_hat

    def evaluate(self, f1_averages:list[str] = ['micro', 'macro', 'weighted']):
        assert (self.alma_sdg.handle == self.predictions.handle).all()

        predictions = self.prepare_tool_labels()
        return super().evaluate(predictions, f1_averages, self.model_name)
    
class ElsevierMulti_eval(Eval):

    def __init__(self):
        super().__init__()
        self.predictions = pd.read_csv(os.path.join(os.pardir,'validation_results', 'elsevier-sdg-multi.csv')).sort_values(by='handle').reset_index(drop=True)
        self.model_name = 'Elsevier-Multilabel'

    def get_multilabel_prediction(self, threshold:float= 0.1):
        
        hat_sdg = super().get_sdg_columns(self.predictions)
        y_hat = self.transform_probability(hat_sdg, threshold)

        return y_hat.to_numpy()

    def transform_probability(self, df:pd.DataFrame, threshold:float) -> pd.DataFrame:
        return df.map(lambda x: True if x>= threshold else False)
   
    def evaluate(self, f1_averages:list[str] = ['micro', 'macro', 'weighted']):
        assert (self.alma_sdg.handle == self.predictions.handle).all()

        predictions = self.get_multilabel_prediction()
        return super().evaluate(predictions, f1_averages, self.model_name)

class ElsevierSingle_eval(Eval):

    def __init__(self):
        super().__init__()
        self.predictions = pd.read_csv(os.path.join(os.pardir, 'validation_results', 'elsevier-sdg-multi.csv')).sort_values(by='handle').reset_index(drop=True)
        self.model_name= 'ElsevierSingle'
    
    def get_singlelabel_prediction(self):
        hat_sdg = self.get_sdg_columns(self.predictions)        
        pred = np.zeros_like(hat_sdg)
        pred[np.arange(hat_sdg.shape[0]), np.nanargmax(hat_sdg, axis = 1)] = 1 

        return pred

    def transform_probability(self, df:pd.DataFrame, threshold:float) -> pd.DataFrame:
        return df.map(lambda x: True if x>= threshold else False)

    def evaluate(self, f1_averages:list[str] = ['micro', 'macro', 'weighted']):
        assert (self.alma_sdg.handle == self.predictions.handle).all()

        predictions = self.get_singlelabel_prediction()
        return super().evaluate(predictions, f1_averages, self.model_name)
    
class Aurora_eval(Eval):

    def __init__(self):
        super().__init__()
        self.model_name = 'Aurora'
        self.predictions = pd.read_csv(os.path.join(os.pardir,  'validation_results', 'aurora-sdg-multi.csv')).sort_values(by='handle').reset_index(drop=True)
    
    def transform_probability(self, df:pd.DataFrame, threshold:float) -> pd.DataFrame:
        return df.map(lambda x: True if x>= threshold else False)
    
    def binarize_predictions(self):
        hat_sdg = super().get_sdg_columns(self.predictions)
        hat_sdg = self.transform_probability(hat_sdg, 0.5)

        return hat_sdg.to_numpy()

    def evaluate(self, f1_averages:list[str] = ['micro', 'macro', 'weighted']):
        assert (self.alma_sdg.handle == self.predictions.handle).all()
        
        predictions = self.binarize_predictions()
        return super().evaluate(predictions, f1_averages, self.model_name)

class GPT_eval(Eval):

    def __init__(self):
        super().__init__()
        self.model_name = 'gpt'
        self.predictions = pd.read_excel(os.path.join(os.pardir,  'validation_results', 'gpt_answers_20082024_t00.xlsx'), sheet_name='ProcessedOutput').sort_values(by='handle').reset_index(drop=True)
    
    def evaluate(self):
        assert (self.alma_sdg.handle == self.predictions.handle).all()

        predictions = self.predictions
        f1_averages = ['micro', 'macro', 'weighted']
        model_name = self.model_name
        return super().evaluate(predictions, f1_averages, model_name)
    
class Gemini_eval(Eval):

    def __init__(self):
        super().__init__()
        self.model_name = 'gemini'
        self.predictions = pd.read_excel(os.path.join(os.pardir,  'validation_results', 'gemini_results_temp00_270824.xlsx'), sheet_name='ProcessedOutput').sort_values(by='handle').reset_index(drop=True)
        # self.alma_sdg = self.alma_sdg.sort_values(by='handle').reset_index(drop=True)
    
    def evaluate(self):
        assert (self.alma_sdg.handle == self.predictions.handle).all()

        predictions = self.predictions
        f1_averages = ['micro', 'macro', 'weighted']
        model_name = self.model_name
        return super().evaluate(predictions, f1_averages, model_name)
    
class LLama_eval(Eval):

    def __init__(self):
        super().__init__()
        self.model_name = 'llama'
        self.predictions = pd.read_excel(os.path.join(os.pardir,  'validation_results', 'llama_01_05_25.xlsx'), sheet_name='ProcessedOutput').sort_values(by='handle').reset_index(drop=True)
        # self.alma_sdg = self.alma_sdg.sort_values(by='handle').reset_index(drop=True)


    def evaluate(self):
        assert (self.alma_sdg.handle == self.predictions.handle).all()

        predictions = self.predictions
        f1_averages = ['micro', 'macro', 'weighted']
        model_name = self.model_name
        return super().evaluate(predictions, f1_averages, model_name)

if __name__ == '__main__':

    other_tools = {'osdg': OSDG_eval(), 
                'elsevierSingleLabel' : ElsevierSingle_eval(),
                'elsevierMultiLabel' : ElsevierMulti_eval(),
                'aurora' : Aurora_eval(),
                'gpt':GPT_eval(),
                'gemini' :  Gemini_eval(),
                'llama' :  LLama_eval()}

    for model_name, model in other_tools.items():
        model.evaluate()
        print('\n')
