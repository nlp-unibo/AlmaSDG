import os
import pandas as pd
import krippendorff
import numpy as np


from itertools import combinations
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from typing import Dict

folder_path = os.path.join(os.pardir, 'datasets', 'annotations')

ANNOTATION_GROUPS = {'Gruppo_1': {'sdg' : ['sdg 01', 'sdg 02', 'sdg 03', 'sdg 08'], 
                                      'annotators': ['Annotator0', 'Annotator1',"gemini", "gpt", "llama"]},
                    'Gruppo_2': {'sdg': ['sdg 06', 'sdg 07', 'sdg 09', 'sdg 11'],
                                    'annotators': ['Annotator3', 'Annotator2',"gemini", "gpt",  "llama"]},
                    'Gruppo_3': {'sdg':['sdg 04', 'sdg 05', 'sdg 10', 'sdg 16'], 
                                'annotators': ['Annotator4', 'Annotator5',"gemini", "gpt", "llama"]},
                    'Gruppo_4': {'sdg':['sdg 12', 'sdg 13', 'sdg 14', 'sdg 15', 'sdg 17'], 
                                    'annotators': ['Annotator6', 'Annotator7',"gemini", "gpt", "llama"]}}

ALL_ANNOTATORS = ['Annotator0', 
                  'Annotator1', 
                  'Annotator2',
                  'Annotator3',
                  'Annotator4', 
                  'Annotator5', 
                  'Annotator6',
                  'Annotator7',
                  "gemini",
                  "gpt",
                  ]


SDG = ['sdg 01', 'sdg 02', 'sdg 03', 'sdg 08', 'sdg 06', 'sdg 07', 'sdg 09', 'sdg 11', 'sdg 04', 'sdg 05', 'sdg 10', 'sdg 16', 'sdg 12', 'sdg 13', 'sdg 14', 'sdg 15', 'sdg 17']
ANNOTATORS = ['Annotator0', 'Annotator1', 'Annotator2', 'Annotator3','Annotator4', 'Annotator5', 'Annotator6', 'Annotator7']
LLM_VERSIONS= {"gemini": {"temp_00": "gemini_results_temp00_270824"},

               "gpt": {"temp_00": "gpt_answers_20082024_t00"},
                         
                "llama": {"temp_00": "llama_01_05_25"}}

abb = {annotator:annotator.split('_')[0] for annotator in ANNOTATORS} | {llm.split('-')[0]:llm.split('-')[0] for llm in LLM_VERSIONS.keys()}
anonymize_list = {name: "Annotator" + str(i) for i,name in enumerate(ANNOTATORS)} | {llm.split('-')[0]:llm.split('-')[0] for llm in LLM_VERSIONS.keys()}#A1-A2... for human annotators

DIRECTORIES = ['completion_tranche', 'relevant_tranche']

def clean_annotation(x:str)-> str:
    return x.replace('sì','si').replace('si ', 'si, ').lower()

def standardize_annotation(labels: pd.Series, annotator:str, llms_names)-> pd.Series:

    if annotator not in llms_names: #can be change to use the data type as a discriminant
        return labels.apply(lambda x: annotation_format[(clean_annotation(x).split(',')[0].strip())])
    
    return labels

def retieve_handles(path: str)-> pd.Series:

    return pd.read_excel(path)['handle']

def convert_to_numeric_label(item_labels: pd.Series, label_format: Dict[str,bool])-> frozenset:
    
    item_labels = dict(item_labels)
    
    if type(list(item_labels.values())[0])== bool:
        labels = frozenset([int(x[-2:]) for x,y in item_labels.items() if y])
    else:
        labels = frozenset([int(x[-2:]) for x,y in item_labels.items() if label_format[clean_annotation(y).split(',')[0]]])

    if labels == frozenset():
        return frozenset([0])

    return labels

def sort_dict(dictionary : Dict) -> Dict:
    return {k: dictionary[k] for k in sorted(dictionary.keys())}

def get_sparse_label_array( df: pd.DataFrame ):
        sdg_labels = df.sdg.values - 1
        label_array = np.zeros((sdg_labels.size, 17))
        label_array[np.arange(sdg_labels.size), sdg_labels] = 1

        return label_array

def latexify(input_list:list)-> str:
   return ' & '.join([str(np.round(value,2)) for value in input_list])

def agreement_llm_humans( annotations, llm_name: str, metric:str ='cohen'):

    other_llms = list(LLM_VERSIONS.keys())
    other_llms.remove(llm_name)

    x = [(annotations[i][metric][k]) for i in ANNOTATION_GROUPS.keys() for k in annotations[i][metric].keys() if llm_name in k and all(other_llm not in k for other_llm in other_llms)] #reliant on names
    llm_human_agreement = pd.DataFrame(x)
    
    sorted_mean_agreement =sort_dict(llm_human_agreement.mean())
    # print(sorted_mean_agreement)
    #for sdg,value in sort_dict(x.mean).items(): print(sdg,'\t', values)
    assert len(sorted_mean_agreement) == llm_human_agreement.shape[1]
    assert list(sorted_mean_agreement.keys()) == sorted(list(sorted_mean_agreement.keys()))

    sorted_values = list(sorted_mean_agreement.values())
    # latex_llm_human_agreement = ' & '.join([str(np.round(value,2)) for value in sorted_values])
    latex_llm_human_agreement = latexify(sorted_values)
    
    print(f'\nHuman-{llm_name} agreement:\n\n{latex_llm_human_agreement}')

    mean_agreement, std_agreement = np.round(np.mean(sorted_values),2), np.round(np.std(sorted_values),2)

    print(f'Mean agreement: {mean_agreement}\tSTD agreement: {std_agreement}\n')

def agreement_llm_llm( annotations, llm_names: list[str], metric:str ='cohen'):

    x = [(annotations[i][metric][k]) for i in ANNOTATION_GROUPS.keys() for k in annotations[i][metric].keys() if all(llm_name in k for llm_name in llm_names)] #reliant on names
    llms_agreement = pd.DataFrame(x)
    
    sorted_mean_agreement =sort_dict(llms_agreement.mean())
    # print(sorted_mean_agreement)
    #for sdg,value in sort_dict(x.mean).items(): print(sdg,'\t', values)
    assert len(sorted_mean_agreement) == llms_agreement.shape[1]
    assert list(sorted_mean_agreement.keys()) == sorted(list(sorted_mean_agreement.keys()))

    sorted_values = list(sorted_mean_agreement.values())
    # latex_llm_human_agreement = ' & '.join([str(np.round(value,2)) for value in sorted_values])
    latex_llm_human_agreement = latexify(sorted_values)
    
    print(f'\n{llm_names[0]}-{llm_names[1]} agreement:\n\n{latex_llm_human_agreement}')

    mean_agreement, std_agreement = np.round(np.mean(sorted_values),2), np.round(np.std(sorted_values),2)

    print(f'Mean agreement: {mean_agreement}\tSTD agreement: {std_agreement}\n')


if __name__ == '__main__':

    human_annotation_path = os.path.join(os.pardir,'datasets', "annotations")
    llm_annotation_path = os.path.join(os.pardir, 'datasets', 'other_tools')

    rounder = lambda x : np.round(x, 2)
    prepare_tool_labels = lambda x: tuple(map(float, str(x).replace('nan', '0').split(',')))

    llm_version = 'temp_00' 
    verbose = True

    agreement_metrics = ['cohen', 'krippendorff']

    human_annotations = ([os.path.join(human_annotation_path, file) for file in os.listdir(human_annotation_path)])
    llm_annotations = [os.path.join(llm_annotation_path, LLM_VERSIONS[file][llm_version])  for file in LLM_VERSIONS.keys()]

    annotation_sheets = ({annotator :  pd.read_excel(os.path.join(human_annotation_path, annotator+'.xlsx')) for annotator in ANNOTATORS}| #Flexible version
                        {model : pd.read_excel(path+'.xlsx', sheet_name = 'ProcessedOutput') for model,path in zip(LLM_VERSIONS.keys(),llm_annotations)})

    for sheet in annotation_sheets.values(): sheet.rename(columns={col: col.strip() for col in sheet.columns}, inplace=True)

    annotation_format = {'no': False,
                        'si': True}
    reversed_annotation_format = {v:k for k,v in annotation_format.items()}
    annotation_values = {'agreements' : {agr_metric: {} for agr_metric in agreement_metrics}} #initialize dict for annotators agreement

    human_agreement = {'cohen': {}, 'krippendorff': {}}
    # for each group we want to compute the agreement between the annotators overall and for each sdg
    for group in ANNOTATION_GROUPS.keys():
        annotation_values[group] = {} #init group's annotation
        annotation_values[group]['cohen'] = {}
        annotation_values[group]['krippendorff'] = {}

        annotators_pairs = combinations(ANNOTATION_GROUPS[group]['annotators'], 2) #generates all the possible annotators pairs [IT'S AN ITERATOR]
        edge_handles = []

        for annotators_pair in annotators_pairs: #Checking pairwise agreement         

            pair_name = '-'.join(annotators_pair)
            save_edge_cases = False

            if annotators_pair[0] not in annotation_sheets.keys() or annotators_pair[1] not in annotation_sheets.keys():
                # if one of the annotators is missing, skip to the next one
                continue

            annotation_values['agreements']['cohen'][pair_name]={}
            annotation_values['agreements']['krippendorff'][pair_name]={}

            annotation_values[group]['cohen'][pair_name]={}
            annotation_values[group]['krippendorff'][pair_name]={}

            pair_annotations = pd.merge(annotation_sheets[annotators_pair[0]],
                                        annotation_sheets[annotators_pair[1]], 
                                        on='handle', suffixes=annotators_pair)

            pair_annotations.index = pair_annotations['handle']
        

            for sdg in ANNOTATION_GROUPS[group]['sdg']: #Gather annotators agreement at SDG level
                human_pair = False

                if annotators_pair[0] not in LLM_VERSIONS.keys() and annotators_pair[1] not in LLM_VERSIONS.keys(): 
                    human_pair = True

                #transform into boolean the human annotations
                annotator_0_labels = standardize_annotation(pair_annotations[sdg + annotators_pair[0]], annotators_pair[0], LLM_VERSIONS.keys())
                annotator_1_labels = standardize_annotation(pair_annotations[sdg + annotators_pair[1]], annotators_pair[1], LLM_VERSIONS.keys())
                assert (annotator_1_labels.index == annotator_0_labels.index).all()

                label_discrepancies = annotator_0_labels != annotator_1_labels # boolean mask for conflict cases

                annotation_values[group][sdg[:6] + ' agreement'] = rounder(cohen_kappa_score(annotator_0_labels, annotator_1_labels))
                annotation_values['agreements']['cohen'][pair_name][int(sdg[4:6].strip())] = (rounder(cohen_kappa_score(annotator_0_labels, annotator_1_labels)))
                annotation_values['agreements']['krippendorff'][pair_name][int(sdg[4:6].strip())] = rounder(krippendorff.alpha(np.array([list(map(lambda x: reversed_annotation_format[x],(annotator_0_labels.values))), 
                                                                                                                            list(map(lambda x: reversed_annotation_format[x],(annotator_1_labels.values)))]).reshape(2,-1), level_of_measurement = 'nominal'))

                annotation_values[group]['cohen'][pair_name][int(sdg[4:6].strip())] = annotation_values['agreements']['cohen'][pair_name][int(sdg[4:6].strip())]
                annotation_values[group]['krippendorff'][pair_name][int(sdg[4:6].strip())] = annotation_values['agreements']['krippendorff'][pair_name][int(sdg[4:6].strip())]

                if verbose:
                    print(f"\nCohen Agreement on {sdg} between {pair_name}:  {annotation_values[group][sdg[:6] + ' agreement']}\n")
                    print(confusion_matrix(annotator_0_labels, annotator_1_labels))
            
                if human_pair:
                    human_agreement['cohen'][sdg] = annotation_values['agreements']['cohen'][pair_name][int(sdg[4:6].strip())]
                    human_agreement['krippendorff'][sdg] = annotation_values['agreements']['krippendorff'][pair_name][int(sdg[4:6].strip())]


    human_agreement['cohen'] = sort_dict(human_agreement['cohen'])
    human_agreement['krippendorff'] = sort_dict(human_agreement['krippendorff'])


    mean_human_agreement, std_human_agreement = np.mean(list(human_agreement['cohen'].values())), np.std(list(human_agreement['cohen'].values()))
    print(f'\nHuman-Human agreement:\n\n{latexify(list(human_agreement["cohen"].values()))}\nMean agreement: {rounder(mean_human_agreement)}\tSTD agreement: {rounder(std_human_agreement)}')


    llm_names = list(LLM_VERSIONS.keys())

    for llm in llm_names:
        agreement_llm_humans(annotation_values, llm)

    for llm_pair in combinations(llm_names, 2): ## Compute LLM-LLM agreement
        agreement_llm_llm(annotation_values, llm_pair)