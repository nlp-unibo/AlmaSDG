import requests
import pandas as pd
import json
import os

from tqdm import tqdm
from typing import List

BASE_URL = "https://aurora-sdg.labs.vu.nl/classifier/classify/"

AVAILABLE_MODELS = ['aurora-sdg-multi', 'elsevier-sdg-multi']

SDG_NAMES = ['sdg 0' + str(i) for i in range(1,10)] + ['sdg ' + str(i) for i in range(10,18)]

predictions = {model_name: {name:[] for name in SDG_NAMES} for model_name in AVAILABLE_MODELS}

def get_data_to_classify(path:str,)-> pd.DataFrame:
    return pd.read_csv(path)

def get_text(data: pd.Series, textual_columns:List[str])-> pd.Series:
    """Retrieve the text from the textual columns and join it"""
    
    return ' '.join(data.loc[textual_columns])

if __name__ == '__main__':
    
    path = os.path.join(os.pardir, 'datasets','alma-sdg.csv')
    data = get_data_to_classify(path=path)
    textual_columns = ['title', 'abstract', 'keywords']
    headers = {'Content-Type': 'application/json'}


    for model in AVAILABLE_MODELS:
        print(f'\nQuerying {model} ...')
        for i,entry in tqdm(data.iterrows(), desc="Articles", total=(len(data))):
            payload = json.dumps({"text": get_text(entry, textual_columns= textual_columns)})
            response = requests.request("POST", BASE_URL + model, headers=headers, data=payload)

            response = json.loads(response.text)
            for idx, j in enumerate(response['predictions']):
                predictions[model][SDG_NAMES[idx]].append(j['prediction'])

        df = pd.DataFrame.from_dict(predictions[model], orient='index').transpose()
        predicts = data[['handle']+ textual_columns].join(df)

        predicts.to_csv(os.path.join(os.pardir, 'datasets', 'other_tools', model + '.csv'), index=False)
