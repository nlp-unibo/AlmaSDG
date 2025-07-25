from openai import OpenAI
import openai
import os
import json
from pypdf import PdfReader
from typing import List,Dict, Optional, Union
import pandas as pd
import numpy as np
import time
from tqdm import tqdm

class GPTQuerier:
  
  def __init__(self, folder_path, dataset_name, sheet_name, seed:int = 24, temperature:float=0.0, prompt_version:str = 'guidelines'):

    self.seed= seed
    self.temperature = temperature
    self.prompt_version = prompt_version

    # self.results = {'SDG '+ str(i):[] for i in range(1,18)}

    self.data_to_query = self.get_dataset(folder_path, dataset_name, sheet_name)
    self.guidelines = self.get_guidelines()
    self.prompt_function = {'guidelines': lambda x,y,z,w: self.generate_prompt_with_guidelines(x, self.guidelines,y,z,w),}
    
    self.label_mapping = {'contributes': True,
                          'does not contribute': False}
    
        
    with open('./gpt_config.json') as f: #Check filepath and the fields
      config = json.load(f)
      openai_api_key = config['openai_api_key']

    self.client = OpenAI(api_key= openai_api_key)

  def get_sdg_description(self)-> Dict:
    sdg_descriptions = ['End poverty in all its forms everywhere',
                      'End hunger, achieve food security and improved nutrition and promote sustainable agriculture',
                      'Ensure healthy lives and promote well-being for all at all ages',
                      'Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all',
                      'Achieve gender equality and empower all women and girls',
                      'Ensure availability and sustainable management of water and sanitation for all',
                      'Ensure access to affordable, reliable, sustainable and modern energy for all',
                      'Promote sustained, inclusive and sustainable economic growth, full and productive employment and decent work for all',
                      'Build resilient infrastructure, promote inclusive and sustainable industrialization and foster innovation',
                      'Reduce inequality within and among countries',
                      'Make cities and human settlements inclusive, safe, resilient and sustainable',
                      'Ensure sustainable consumption and production patterns',
                      'Take urgent action to combat climate change and its impacts',
                      'Conserve and sustainably use the oceans, seas and marine resources for sustainable development',
                      'Protect, restore and promote sustainable use of terrestrial ecosystems, sustainably manage forests, combat desertification, and halt and reverse land degradation and halt biodiversity loss',
                      'Promote peaceful and inclusive societies for sustainable development, provide access to justice for all and build effective, accountable and inclusive institutions at all levels',
                      'Strengthen the means of implementation and revitalize the Global Partnership for Sustainable Development']

    return  {i+1:des for i,des in enumerate(sdg_descriptions)}

  def get_pdf_guidelines(self, path:str=None)-> PdfReader:

    if path == None:

      path = os.path.join(os.pardir, 'dataset', 'guidelines.pdf')

    reader = PdfReader(path)

    return reader

  def get_guidelines(self)-> Dict[str,Dict[str,str]]:
    sdg_descriptions = self.get_sdg_description()
    reader = self.get_pdf_guidelines()

    sdg_guidelines = {'SDG ' + str(i): {'description': sdg_descriptions[i],
                                        'guidelines': (reader.pages[i+2].extract_text().replace(' , ', ', ').replace(' . ', '. ').replace('  ', ' ').replace('Index', '').strip())}
                      for i in range(1,18)}
    
    return sdg_guidelines

  def get_dataset(self, folder_path: str, input_file:str = 'complete_dataset.xlsx', sheet_name: str ='ArticlesToTag')->pd.DataFrame:
    
    if type(folder_path) == list:
      file_path = os.path.join(*folder_path, input_file)
    else: 
      file_path = os.path.join(folder_path, input_file)
    
    if input_file.split('.')[-1] == 'xlsx':
      return pd.read_excel(file_path, sheet_name=sheet_name)
    else:
      return pd.read_csv(file_path)

  def generate_prompt_with_guidelines(self, sdg:str, sdg_guidelines:Dict[str,Dict[str,str]], title:str, abstract:str, keywords:str) -> str:
    prompt_model = f"""You are an expert in sustainable development. You must determine if a given scientific article contributes to {sdg}: {sdg_guidelines[sdg]['description']}

                      You must base your decision on the article's TITLE, ABSTRACT, and KEYWORDS, and on the following GUIDELINES:

                      GUIDELINES:
                      The purpose of the study is key to discriminate the themes the paper
                      contributes to (CONTRIBUTES) from those the paper does not contribute to but mentions
                      them as part of the general background context (DOES NOT CONTRIBUTE).

                      Possible impacts or implications relevant to an SDG should be considered only if they
                      are stated explicitly in the document.

                      {sdg_guidelines[sdg]['guidelines']}

                      Your output should be either "CONTRIBUTES" or "DOES NOT CONTRIBUTE".
                      You should also explain your output based on the GUIDELINES.

                      TITLE:	{title}
                      ABSTRACT: {abstract}
                      KEYWORDS: {keywords}
                      OUTPUT:
                      """
    
    return prompt_model

  def process_answers(self, answers:pd.DataFrame)-> pd.DataFrame:
    ## Save all the explanations, ease readability
    sdg_labels = sorted([col for col in answers.columns if col.lower().startswith('sdg')],
                            key=lambda x: int(x.split('sdg')[1]))
    
    processed_answers = answers[sdg_labels].map(lambda x: self.label_mapping[x.split('\n')[0].replace('*','').replace('OUTPUT:', '').strip().lower()], na_action='ignore')

    return processed_answers

  def answer_polarity(self, text:str):
  
    return self.label_mapping[text.split('\n')[0].replace('*','').replace('OUTPUT:', '').strip().lower()]

  def save_results(self, results: pd.DataFrame, original_data:pd.DataFrame,path:str, start_index:int = 0, output_file:str = '')-> None:
    
    frame_results = pd.DataFrame.from_dict(results, orient='index').transpose()
    processed_results = self.process_answers(frame_results)
    
    try:
      data_to_add = original_data.drop(columns= 'comments')
    except KeyError:
      data_to_add = original_data
      
    size = frame_results.shape[0]

    if 'title' not in frame_results.columns:
      results_with_text = data_to_add.iloc[start_index:start_index+size,:].join(frame_results)
      processed_with_text = data_to_add.iloc[start_index:start_index+size,:].join(processed_results)

    if os.path.isfile(output_file):
      writer = pd.ExcelWriter(path, mode= 'a', if_sheet_exists='new')
    else:
      writer = pd.ExcelWriter(path, mode= 'w')

    with writer as writer:
      results_with_text.to_excel(writer, sheet_name='GPTAnswers', index=False)
      processed_with_text.to_excel(writer, sheet_name='ProcessedOutput', index=False)

    return None

  def gpt_call(self, prompt_model:str):
    response = self.client.chat.completions.create(model="gpt-4o-mini",
                                                  messages=[{"role": "system", "content": "You are a helpful expert in sustainable development."},
                                                            {"role": "user", "content": prompt_model}],
                                                  seed= self.seed,
                                                  temperature= self.temperature)
    
    return response

  def query_sdg(self, sdg:int, title, abstract, keywords):
    """Query ChatGPT 4o-mini with the class prompt template for a specific SDG"""

    current_sdg = 'SDG '+ str(sdg)
    prompt_model = self.prompt_function[self.prompt_version](current_sdg, title, abstract, keywords)

    try:
      response = self.gpt_call(prompt_model)
        
    except openai.RateLimitError: #Too many calls in one minute, stopping for a little while
      print('Too many reqeusts in current time window, pausing for 30 sec.')
      time.sleep(60) #exponential backoff suggested in openai site
      response = self.gpt_call(prompt_model)

    return self.answer_polarity(response.choices[0].message.content), response.choices[0].message.content

  def exception_save(self, results, output_file, already_queried:int = 0):
        print('Something went wrong, saving results...')
        self.save_results(results, self.data_to_query, path=output_file, start_index=already_queried)

dataset_name, sheet_name = 'alma_sdg.csv', 'None'

input_path =   [os.pardir, 'dataset']
output_path = [os.pardir, 'validation_results']

ANNOTATION_FOLDER_PATH = os.path.join(*output_path)

debug = False
name= 'gpt_validation_results.xlsx'
output_file = os.path.join(ANNOTATION_FOLDER_PATH, name) #output_file

results = {'handle':[], 'title':[], 'abstract': [], 'keywords':[]} | {'sdg ' + str(i):[] for i in range(1,18)}

querier = GPTQuerier(input_path, dataset_name, sheet_name)


for index in tqdm(range(querier.data_to_query.shape[0]), desc='Article', position= 0, leave = True):# Loop for each article in the dataset

  adapted_index = index  
  title = querier.data_to_query.iloc[adapted_index].title
  abstract = querier.data_to_query.iloc[adapted_index].abstract
  keywords = querier.data_to_query.iloc[adapted_index].keywords
  handle = querier.data_to_query.iloc[adapted_index].handle
  
  # if index % 50 == 0:
  #   with pd.ExcelWriter(output_file) as writer:
  #       frame_results = pd.DataFrame.from_dict(querier.results, orient='index').transpose()
        
  #       frame_results.to_excel(writer, sheet_name='UnprocessedAnswers')
  #       processed_frame = querier.process_answers(frame_results)
  #       querier.save_results(frame_results, processed_frame, querier.data_to_query, output_file, start_index=already_queried)

  for sdg in tqdm(range(1,18), desc='SDG', leave=False): #Loop for each SDG ... Turn into a method? #Change to match specific SDGs

    try: ### Methodizable
      answer_polarity, answer = querier.query_sdg(sdg, title, abstract, keywords)
    except Exception as e:
      querier.exception_save(results, output_file=output_file)
      raise e


querier.save_results(results, querier.data_to_query, path= output_file)



print('Successfully queried all articles. Closing...')