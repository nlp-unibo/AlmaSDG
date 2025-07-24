!pip install transformers
!pip install bitsandbytes
!pip install accelerate
!pip install torch

import transformers
import torch
import pandas as pd

from google.colab import userdata

model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
access_token = userdata.get('Huggingface')

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    token=access_token,
    model_kwargs={"torch_dtype": torch.bfloat16, "load_in_4bit": True}
)

df = pd.read_excel("sdg.xlsx")
df.head()

for i in range(0, len(df)):
  content = "Title:\n" + df['title'][i] + "\nAbstract:\n" + df['abstract'][i] + "\nKeywords:\n" + df['keywords'][i]

  messages = [
    {"role": "system", "content": """You are an helpful and honest Sustainable Development Goals (SDGs) expert. Your job is to state whether or not the paper contributes to SDG 1: End poverty in all its forms everywhere. Your answer should be a simple "yes" or "no".

    These are the guidelines:
    How does a paper “contribute to” an SDG?
    A paper can advance sustainable development in many ways: by applying existing technologies, introducing new technologies or methods, examining case studies, or reviewing them to highlight criticalities and possible improvements.
    Importantly, to qualify as a “contribution”, a paper should address an aspect of the SDG's overall theme. These aspects/areas are defined later for each SDG.
    The relation between paper and aspect of the SDG theme could be explicit or implicit: a paper does not necessarily have to explicitly declare the intent of contributing to the SDG (e.g., our work aims at improving sustainable agriculture as described in SDG 2). The relation can also be established based on the paper’s output and the areas of impact. Moreover, the purpose of the study is key to further discriminate the themes the paper contributes to (contribution) from those the paper does not contribute to but mentions them as part of the general background context (no contribution).

    To what extent should the annotator leverage their own knowledge/technical expertise and consider the implicit outcomes or implications of the paper?
    Possible impacts or implications relevant to an SDG should be considered only if they are stated explicitly in the document. For instance, if an article enhances a waste processing system, even if the annotator infers a reduction of energy consumption, the paper should not be tagged with SDG 7 (clean energy) unless the paper makes such a connection explicit.
    Does the paper have to provide a direct/measurable advancement in terms of SDG?

    Not necessarily. Indirect advancements can represent a contribution. These include:
    - surveys on various methods or the current state of the art
    - research producing resources as foundations for future developments
    - enhancements of evaluation methods
    For example, the production of a dataset gathering information about marine pollution to simplify the development of cleaning methods or provide information for sustainable policies and strategies implementation represents a contribution to SDG 14 (life below water).

    How to discriminate between all candidate SDGs that seem related to the paper?
    The SDGs encompass a wide variety of heavily related themes, tackling them from different angles. This might induce the annotator to tag an article with a set of SDGs sharing the same themes or subgoals. However, in choosing the relevant SDGs, it is important to distinguish those mentioned as background context (no contribution) from those where the aim of the study is positioned in terms of sustainable approaches (contribution).
    For example, an article assessing the people’s preparation and information on climate change could be interpreted as a contribution to SDG 4 (quality education), SDG 12 (responsible production and consumption), and SDG 13 (climate action) which share the theme of sustainable knowledge promotion. However, only the latter would be appropriate in this instance because SDG 4 is about scholastic and academic contexts and SDG 12 is about sustainable lifestyles, consumption and production patterns. Another example is the connection between SDG 5 and SDG 10. Both aim to reduce inequalities. Goal 10 tackles discrimination and inclusion in a broader sense, encompassing economic, social, and political aspects and migratory phenomena. Goal 5 shares the same equality objective but is focused on gender equality and its challenges. Articles tackling these issues from both angles (general and gender-specific) should be tagged with both SDGs; otherwise, the purpose of the paper should determine which SDG is appropriate.

    Is it necessary to read/understand all the areas/aspects of all SDGs?
    Not all the aspects, but when deciding about the contribution of a paper to a given SDG, it is important to be aware of other related SDGs, because this helps understand the scope and boundaries of the SDG under consideration.
    If the annotator is not an SDG expert, they should read the first paragraph of each SDG description card (for example, to know that SDG 2 about hunger, sustainable agriculture, and access to food). When in doubt, they can refer to the detailed description/areas.


    SDG 1: End poverty in all its forms everywhere
    Articles contribute to SDG 1 if they are about poverty. For example, articles that mention or may have an impact on the following areas should be considered relevant:
    - Poverty
    - Social protection systems/measures
    - Access/right of all people to economic resources, basic services (such as energy and water services)
    - Rights to ownership and control over land and other forms of property, inheritance, natural resources, technology and financial services, including microfinance
    - Resilience/exposure of the poor/vulnerable to extreme events and other economic, social and environmental shocks and disasters
    - Cooperation resources for developing countries

    Example:
    Articles that study or enable access to safe drinking water for vulnerable segments of population, enhance access to energy reducing energy poverty, introduce or examine welfare measures for minimum wages are relevant to SDG 1 because they aim at reducing different forms of poverty.
    Articles that study efficient energy production systems, oriented to reduce energy consumption-related costs, without tying it to energy poverty concerns are not relevant to SDG 1.

  """},


  {"role": "user", "content": content},
  ]

  prompt = pipeline.tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
  )

  terminators = [
      pipeline.tokenizer.eos_token_id,
      pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  outputs = pipeline(
      prompt,
      max_new_tokens=1000,
      eos_token_id=terminators,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
  )

  print(outputs[0]["generated_text"][len(prompt):])

#SDG 2
answers = []

for i in range(0, len(df)):
  content = "Title:\n" + df['title'][i] + "\nAbstract:\n" + df['abstract'][i] + "\nKeywords:\n" + df['keywords'][i]

  messages = [
    {"role": "system", "content": """You are an helpful and honest Sustainable Development Goals (SDGs) expert. Your job is to state whether or not the paper contributes to SDG 2: End hunger, achieve food security and improved nutrition and promote sustainable agriculture. Your answer should be a simple "yes" or "no".

    These are the guidelines:
    How does a paper “contribute to” an SDG?
    A paper can advance sustainable development in many ways: by applying existing technologies, introducing new technologies or methods, examining case studies, or reviewing them to highlight criticalities and possible improvements.
    Importantly, to qualify as a “contribution”, a paper should address an aspect of the SDG's overall theme. These aspects/areas are defined later for each SDG.
    The relation between paper and aspect of the SDG theme could be explicit or implicit: a paper does not necessarily have to explicitly declare the intent of contributing to the SDG (e.g., our work aims at improving sustainable agriculture as described in SDG 2). The relation can also be established based on the paper’s output and the areas of impact. Moreover, the purpose of the study is key to further discriminate the themes the paper contributes to (contribution) from those the paper does not contribute to but mentions them as part of the general background context (no contribution).

    To what extent should the annotator leverage their own knowledge/technical expertise and consider the implicit outcomes or implications of the paper?
    Possible impacts or implications relevant to an SDG should be considered only if they are stated explicitly in the document. For instance, if an article enhances a waste processing system, even if the annotator infers a reduction of energy consumption, the paper should not be tagged with SDG 7 (clean energy) unless the paper makes such a connection explicit.
    Does the paper have to provide a direct/measurable advancement in terms of SDG?

    Not necessarily. Indirect advancements can represent a contribution. These include:
    - surveys on various methods or the current state of the art
    - research producing resources as foundations for future developments
    - enhancements of evaluation methods
    For example, the production of a dataset gathering information about marine pollution to simplify the development of cleaning methods or provide information for sustainable policies and strategies implementation represents a contribution to SDG 14 (life below water).

    How to discriminate between all candidate SDGs that seem related to the paper?
    The SDGs encompass a wide variety of heavily related themes, tackling them from different angles. This might induce the annotator to tag an article with a set of SDGs sharing the same themes or subgoals. However, in choosing the relevant SDGs, it is important to distinguish those mentioned as background context (no contribution) from those where the aim of the study is positioned in terms of sustainable approaches (contribution).
    For example, an article assessing the people’s preparation and information on climate change could be interpreted as a contribution to SDG 4 (quality education), SDG 12 (responsible production and consumption), and SDG 13 (climate action) which share the theme of sustainable knowledge promotion. However, only the latter would be appropriate in this instance because SDG 4 is about scholastic and academic contexts and SDG 12 is about sustainable lifestyles, consumption and production patterns. Another example is the connection between SDG 5 and SDG 10. Both aim to reduce inequalities. Goal 10 tackles discrimination and inclusion in a broader sense, encompassing economic, social, and political aspects and migratory phenomena. Goal 5 shares the same equality objective but is focused on gender equality and its challenges. Articles tackling these issues from both angles (general and gender-specific) should be tagged with both SDGs; otherwise, the purpose of the paper should determine which SDG is appropriate.

    Is it necessary to read/understand all the areas/aspects of all SDGs?
    Not all the aspects, but when deciding about the contribution of a paper to a given SDG, it is important to be aware of other related SDGs, because this helps understand the scope and boundaries of the SDG under consideration.
    If the annotator is not an SDG expert, they should read the first paragraph of each SDG description card (for example, to know that SDG 2 about hunger, sustainable agriculture, and access to food). When in doubt, they can refer to the detailed description/areas.


    SDG 2: End hunger, achieve food security and improved nutrition and promote sustainable agriculture
    Articles contribute to SDG 2 if they are about hunger, sustainable agriculture, and access to food. For example, articles that mention or may have an impact on the following areas should be considered relevant:
    - Hunger, malnutrition and food insecurity
    - Access to food
    - Social protection to safeguard safe and nutritious food
    - Transforming food systems for a more inclusive and sustainable world
    - Investments to help poor people have access to food
    - Correction/prevention of trade restrictions and distortions in world agricultural markets; economic/trade aspects of food commodity markets
    - Stunting and wasting in children, nutritional needs of adolescent girls, pregnant and lactating women and older persons
    - Agricultural research, productivity and sustainability of food producers
    - Improving land and soil quality
    - Promoting secure and equal access to land, other productive resources and inputs, knowledge, benefits arising from the utilization of genetic resources, financial services, markets and opportunities
    - Increasing productivity and production, maintaining ecosystems, strengthening resilience to climate change and natural disasters
    - Maintaining genetic diversity

    Example:
    Articles that study the employment of renewable energy supplies in swine farms, or vertical farms in urban areas, are relevant to SDG 2 because they are about sustainable food production.
    Articles that study how to improve flavour of curated cheese, if not explicitly linked to SDG 2, are not relevant to SDG 2 because they do not contribute to hunger, sustainability in agriculture, access to food etc.

  """},


  {"role": "user", "content": content},
  ]

  prompt = pipeline.tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
  )

  terminators = [
      pipeline.tokenizer.eos_token_id,
      pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  outputs = pipeline(
      prompt,
      max_new_tokens=1000,
      eos_token_id=terminators,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
  )

  print(outputs[0]["generated_text"][len(prompt):])
  answers.append(str(outputs[0]["generated_text"][len(prompt):]))

for a in answers:
  print(a)

#SDG 3
answers = []

for i in range(0, len(df)):
  content = "Title:\n" + df['title'][i] + "\nAbstract:\n" + df['abstract'][i] + "\nKeywords:\n" + df['keywords'][i]

  messages = [
    {"role": "system", "content": """You are an helpful and honest Sustainable Development Goals (SDGs) expert. Your job is to state whether or not the paper contributes to SDG 3: Ensure healthy lives and promote well-being fo all at all ages. Your answer should be a simple "yes" or "no".

    These are the guidelines:
    How does a paper “contribute to” an SDG?
    A paper can advance sustainable development in many ways: by applying existing technologies, introducing new technologies or methods, examining case studies, or reviewing them to highlight criticalities and possible improvements.
    Importantly, to qualify as a “contribution”, a paper should address an aspect of the SDG's overall theme. These aspects/areas are defined later for each SDG.
    The relation between paper and aspect of the SDG theme could be explicit or implicit: a paper does not necessarily have to explicitly declare the intent of contributing to the SDG (e.g., our work aims at improving sustainable agriculture as described in SDG 2). The relation can also be established based on the paper’s output and the areas of impact. Moreover, the purpose of the study is key to further discriminate the themes the paper contributes to (contribution) from those the paper does not contribute to but mentions them as part of the general background context (no contribution).

    To what extent should the annotator leverage their own knowledge/technical expertise and consider the implicit outcomes or implications of the paper?
    Possible impacts or implications relevant to an SDG should be considered only if they are stated explicitly in the document. For instance, if an article enhances a waste processing system, even if the annotator infers a reduction of energy consumption, the paper should not be tagged with SDG 7 (clean energy) unless the paper makes such a connection explicit.
    Does the paper have to provide a direct/measurable advancement in terms of SDG?

    Not necessarily. Indirect advancements can represent a contribution. These include:
    - surveys on various methods or the current state of the art
    - research producing resources as foundations for future developments
    - enhancements of evaluation methods
    For example, the production of a dataset gathering information about marine pollution to simplify the development of cleaning methods or provide information for sustainable policies and strategies implementation represents a contribution to SDG 14 (life below water).

    How to discriminate between all candidate SDGs that seem related to the paper?
    The SDGs encompass a wide variety of heavily related themes, tackling them from different angles. This might induce the annotator to tag an article with a set of SDGs sharing the same themes or subgoals. However, in choosing the relevant SDGs, it is important to distinguish those mentioned as background context (no contribution) from those where the aim of the study is positioned in terms of sustainable approaches (contribution).
    For example, an article assessing the people’s preparation and information on climate change could be interpreted as a contribution to SDG 4 (quality education), SDG 12 (responsible production and consumption), and SDG 13 (climate action) which share the theme of sustainable knowledge promotion. However, only the latter would be appropriate in this instance because SDG 4 is about scholastic and academic contexts and SDG 12 is about sustainable lifestyles, consumption and production patterns. Another example is the connection between SDG 5 and SDG 10. Both aim to reduce inequalities. Goal 10 tackles discrimination and inclusion in a broader sense, encompassing economic, social, and political aspects and migratory phenomena. Goal 5 shares the same equality objective but is focused on gender equality and its challenges. Articles tackling these issues from both angles (general and gender-specific) should be tagged with both SDGs; otherwise, the purpose of the paper should determine which SDG is appropriate.

    Is it necessary to read/understand all the areas/aspects of all SDGs?
    Not all the aspects, but when deciding about the contribution of a paper to a given SDG, it is important to be aware of other related SDGs, because this helps understand the scope and boundaries of the SDG under consideration.
    If the annotator is not an SDG expert, they should read the first paragraph of each SDG description card (for example, to know that SDG 2 about hunger, sustainable agriculture, and access to food). When in doubt, they can refer to the detailed description/areas.


    SDG 3: Ensure healthy lives and promote well-being fo all at all ages
    Articles contribute to SDG 3 if they are about improving people’s health. For example, articles that mention or may have an impact on the following areas should be considered relevant:
    - Medicine
    - Diseases, epidemics, mortality, regional prevalence of diseases
    - Prevention and treatments, medicines, vaccines
    - Access to health care, universal health coverage, equitable access, addressing disparities
    - Investment in health systems
    - Resilience against future health threats
    - Environmental and commercial factors affecting health and access to healthcare
    - Increasing awareness of and information about importance of good health, healthy lifestyles, making well-informed choices, practicing safe sex and vaccinating children
    - Promotion of mental health and well-being

    Example:
    Articles that study policies to reduce substance abuse in communities or study the prevalence of death and injuries in road traffic accidents in urban areas are relevant to SDG 3 because they are about improving health and well-being.
    Articles that study efficient packaging systems for tablets are not relevant to SDG 3 because they do not address health and well-being, unless an explicit connection is made to SDG 3, for example in terms of reducing the cost/improving access to treatments.
  """},


  {"role": "user", "content": content},
  ]

  prompt = pipeline.tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
  )

  terminators = [
      pipeline.tokenizer.eos_token_id,
      pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  outputs = pipeline(
      prompt,
      max_new_tokens=1000,
      eos_token_id=terminators,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
  )

  print(outputs[0]["generated_text"][len(prompt):])
  answers.append(str(outputs[0]["generated_text"][len(prompt):]))

print("Answers:\n")
for a in answers:
  print(a)

#SDG 4
answers = []

for i in range(0, len(df)):
  content = "Title:\n" + df['title'][i] + "\nAbstract:\n" + df['abstract'][i] + "\nKeywords:\n" + df['keywords'][i]

  messages = [
    {"role": "system", "content": """You are an helpful and honest Sustainable Development Goals (SDGs) expert. Your job is to state whether or not the paper contributes to SDG 4: Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all. Your answer should be a simple "yes" or "no".

    These are the guidelines:
    How does a paper “contribute to” an SDG?
    A paper can advance sustainable development in many ways: by applying existing technologies, introducing new technologies or methods, examining case studies, or reviewing them to highlight criticalities and possible improvements.
    Importantly, to qualify as a “contribution”, a paper should address an aspect of the SDG's overall theme. These aspects/areas are defined later for each SDG.
    The relation between paper and aspect of the SDG theme could be explicit or implicit: a paper does not necessarily have to explicitly declare the intent of contributing to the SDG (e.g., our work aims at improving sustainable agriculture as described in SDG 2). The relation can also be established based on the paper’s output and the areas of impact. Moreover, the purpose of the study is key to further discriminate the themes the paper contributes to (contribution) from those the paper does not contribute to but mentions them as part of the general background context (no contribution).

    To what extent should the annotator leverage their own knowledge/technical expertise and consider the implicit outcomes or implications of the paper?
    Possible impacts or implications relevant to an SDG should be considered only if they are stated explicitly in the document. For instance, if an article enhances a waste processing system, even if the annotator infers a reduction of energy consumption, the paper should not be tagged with SDG 7 (clean energy) unless the paper makes such a connection explicit.
    Does the paper have to provide a direct/measurable advancement in terms of SDG?

    Not necessarily. Indirect advancements can represent a contribution. These include:
    - surveys on various methods or the current state of the art
    - research producing resources as foundations for future developments
    - enhancements of evaluation methods
    For example, the production of a dataset gathering information about marine pollution to simplify the development of cleaning methods or provide information for sustainable policies and strategies implementation represents a contribution to SDG 14 (life below water).

    How to discriminate between all candidate SDGs that seem related to the paper?
    The SDGs encompass a wide variety of heavily related themes, tackling them from different angles. This might induce the annotator to tag an article with a set of SDGs sharing the same themes or subgoals. However, in choosing the relevant SDGs, it is important to distinguish those mentioned as background context (no contribution) from those where the aim of the study is positioned in terms of sustainable approaches (contribution).
    For example, an article assessing the people’s preparation and information on climate change could be interpreted as a contribution to SDG 4 (quality education), SDG 12 (responsible production and consumption), and SDG 13 (climate action) which share the theme of sustainable knowledge promotion. However, only the latter would be appropriate in this instance because SDG 4 is about scholastic and academic contexts and SDG 12 is about sustainable lifestyles, consumption and production patterns. Another example is the connection between SDG 5 and SDG 10. Both aim to reduce inequalities. Goal 10 tackles discrimination and inclusion in a broader sense, encompassing economic, social, and political aspects and migratory phenomena. Goal 5 shares the same equality objective but is focused on gender equality and its challenges. Articles tackling these issues from both angles (general and gender-specific) should be tagged with both SDGs; otherwise, the purpose of the paper should determine which SDG is appropriate.

    Is it necessary to read/understand all the areas/aspects of all SDGs?
    Not all the aspects, but when deciding about the contribution of a paper to a given SDG, it is important to be aware of other related SDGs, because this helps understand the scope and boundaries of the SDG under consideration.
    If the annotator is not an SDG expert, they should read the first paragraph of each SDG description card (for example, to know that SDG 2 about hunger, sustainable agriculture, and access to food). When in doubt, they can refer to the detailed description/areas.


    SDG 4: Ensure inclusive and equitable quality education and promote lifelong learning opportunities for all
    Articles contribute to SDG 4 if they are about education and improve its quality and accessibility. For example, articles that mention or may have an impact on the following areas should be considered relevant:
    - Universal literacy and numeracy
    - Equal education accessibility at all levels (pre-primary, primary, university...)
    - Promote education on sustainable development and global citizenship
    - Improve the teaching in youths and adults of vocational and technical skills for employment
    - Increase the supply of qualified teachers
    - Improvement of teaching methods
    - Increase the number of inclusive and safe schools
    - Expand scholarships for developing countries

    Example:
    Articles that discuss methods to improve teachers’ preparation or study vulnerable students’ learning patterns are relevant to SDG 4.
    Articles that study the education levels’ effects on people’s well-being, or about improving network security in online teaching, if not explicitly linked to SDG4, are not relevant to SDG 4 because are not related to teaching and education quality.
  """},


  {"role": "user", "content": content},
  ]

  prompt = pipeline.tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
  )

  terminators = [
      pipeline.tokenizer.eos_token_id,
      pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  outputs = pipeline(
      prompt,
      max_new_tokens=1000,
      eos_token_id=terminators,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
  )

  print(outputs[0]["generated_text"][len(prompt):])
  answers.append(str(outputs[0]["generated_text"][len(prompt):]))

print("Answers:\n")
for a in answers:
  print(a)

#SDG 5
answers = []

for i in range(0, len(df)):
  content = "Title:\n" + df['title'][i] + "\nAbstract:\n" + df['abstract'][i] + "\nKeywords:\n" + df['keywords'][i]

  messages = [
    {"role": "system", "content": """You are an helpful and honest Sustainable Development Goals (SDGs) expert. Your job is to state whether or not the paper contributes to SDG 5: Achieve gender equality and empower all women and girls. Your answer should be a simple "yes" or "no".

    These are the guidelines:
    How does a paper “contribute to” an SDG?
    A paper can advance sustainable development in many ways: by applying existing technologies, introducing new technologies or methods, examining case studies, or reviewing them to highlight criticalities and possible improvements.
    Importantly, to qualify as a “contribution”, a paper should address an aspect of the SDG's overall theme. These aspects/areas are defined later for each SDG.
    The relation between paper and aspect of the SDG theme could be explicit or implicit: a paper does not necessarily have to explicitly declare the intent of contributing to the SDG (e.g., our work aims at improving sustainable agriculture as described in SDG 2). The relation can also be established based on the paper’s output and the areas of impact. Moreover, the purpose of the study is key to further discriminate the themes the paper contributes to (contribution) from those the paper does not contribute to but mentions them as part of the general background context (no contribution).

    To what extent should the annotator leverage their own knowledge/technical expertise and consider the implicit outcomes or implications of the paper?
    Possible impacts or implications relevant to an SDG should be considered only if they are stated explicitly in the document. For instance, if an article enhances a waste processing system, even if the annotator infers a reduction of energy consumption, the paper should not be tagged with SDG 7 (clean energy) unless the paper makes such a connection explicit.
    Does the paper have to provide a direct/measurable advancement in terms of SDG?

    Not necessarily. Indirect advancements can represent a contribution. These include:
    - surveys on various methods or the current state of the art
    - research producing resources as foundations for future developments
    - enhancements of evaluation methods
    For example, the production of a dataset gathering information about marine pollution to simplify the development of cleaning methods or provide information for sustainable policies and strategies implementation represents a contribution to SDG 14 (life below water).

    How to discriminate between all candidate SDGs that seem related to the paper?
    The SDGs encompass a wide variety of heavily related themes, tackling them from different angles. This might induce the annotator to tag an article with a set of SDGs sharing the same themes or subgoals. However, in choosing the relevant SDGs, it is important to distinguish those mentioned as background context (no contribution) from those where the aim of the study is positioned in terms of sustainable approaches (contribution).
    For example, an article assessing the people’s preparation and information on climate change could be interpreted as a contribution to SDG 4 (quality education), SDG 12 (responsible production and consumption), and SDG 13 (climate action) which share the theme of sustainable knowledge promotion. However, only the latter would be appropriate in this instance because SDG 4 is about scholastic and academic contexts and SDG 12 is about sustainable lifestyles, consumption and production patterns. Another example is the connection between SDG 5 and SDG 10. Both aim to reduce inequalities. Goal 10 tackles discrimination and inclusion in a broader sense, encompassing economic, social, and political aspects and migratory phenomena. Goal 5 shares the same equality objective but is focused on gender equality and its challenges. Articles tackling these issues from both angles (general and gender-specific) should be tagged with both SDGs; otherwise, the purpose of the paper should determine which SDG is appropriate.

    Is it necessary to read/understand all the areas/aspects of all SDGs?
    Not all the aspects, but when deciding about the contribution of a paper to a given SDG, it is important to be aware of other related SDGs, because this helps understand the scope and boundaries of the SDG under consideration.
    If the annotator is not an SDG expert, they should read the first paragraph of each SDG description card (for example, to know that SDG 2 about hunger, sustainable agriculture, and access to food). When in doubt, they can refer to the detailed description/areas.


    SDG 5: Achieve gender equality and empower all women and girls
    Articles contribute to SDG 5 if they are about gender equality and women empowerment. For example, articles that mention or may have an impact on the following areas should be considered relevant:
    - Discrimination, violence and harmful practices against women and girls
    - Legal frameworks for gender equality and non-discrimination
    - Women in political/managerial positions
    - Recognition and payment unpaid domestic and care work
    - Equal access to sexual and reproductive health care, and empowerment of their own decision-making
    - Female land rights or ownership (land ownership)
    - Promotion of women’s empowerment (through technology, involvement in decision-making processes)
    - Sound policies and enforceable legislation to promote empowerment at all levels

    Example:
    Articles that study women’s political participation, explores Gender Based Violence themes (e.g. revenge porn), or the effects of gender wage gap should be considered relevant to SDG 5.
    Articles that analyse pathologies in female population should not be considered relevant to SDG 5, unless they highlight differences between genders during the treatment or as specific effects.
  """},


  {"role": "user", "content": content},
  ]

  prompt = pipeline.tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
  )

  terminators = [
      pipeline.tokenizer.eos_token_id,
      pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  outputs = pipeline(
      prompt,
      max_new_tokens=1000,
      eos_token_id=terminators,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
  )

  print(outputs[0]["generated_text"][len(prompt):])
  answers.append(str(outputs[0]["generated_text"][len(prompt):]))

print("Answers:\n")
for a in answers:
  print(a)

#SDG 6
answers = []

for i in range(0, len(df)):
  content = "Title:\n" + df['title'][i] + "\nAbstract:\n" + df['abstract'][i] + "\nKeywords:\n" + df['keywords'][i]

  messages = [
    {"role": "system", "content": """You are an helpful and honest Sustainable Development Goals (SDGs) expert. Your job is to state whether or not the paper contributes to SDG 6: Ensure availability and sustainable management of water and sanitation for all. Your answer should be a simple "yes" or "no".

    These are the guidelines:
    How does a paper “contribute to” an SDG?
    A paper can advance sustainable development in many ways: by applying existing technologies, introducing new technologies or methods, examining case studies, or reviewing them to highlight criticalities and possible improvements.
    Importantly, to qualify as a “contribution”, a paper should address an aspect of the SDG's overall theme. These aspects/areas are defined later for each SDG.
    The relation between paper and aspect of the SDG theme could be explicit or implicit: a paper does not necessarily have to explicitly declare the intent of contributing to the SDG (e.g., our work aims at improving sustainable agriculture as described in SDG 2). The relation can also be established based on the paper’s output and the areas of impact. Moreover, the purpose of the study is key to further discriminate the themes the paper contributes to (contribution) from those the paper does not contribute to but mentions them as part of the general background context (no contribution).

    To what extent should the annotator leverage their own knowledge/technical expertise and consider the implicit outcomes or implications of the paper?
    Possible impacts or implications relevant to an SDG should be considered only if they are stated explicitly in the document. For instance, if an article enhances a waste processing system, even if the annotator infers a reduction of energy consumption, the paper should not be tagged with SDG 7 (clean energy) unless the paper makes such a connection explicit.
    Does the paper have to provide a direct/measurable advancement in terms of SDG?

    Not necessarily. Indirect advancements can represent a contribution. These include:
    - surveys on various methods or the current state of the art
    - research producing resources as foundations for future developments
    - enhancements of evaluation methods
    For example, the production of a dataset gathering information about marine pollution to simplify the development of cleaning methods or provide information for sustainable policies and strategies implementation represents a contribution to SDG 14 (life below water).

    How to discriminate between all candidate SDGs that seem related to the paper?
    The SDGs encompass a wide variety of heavily related themes, tackling them from different angles. This might induce the annotator to tag an article with a set of SDGs sharing the same themes or subgoals. However, in choosing the relevant SDGs, it is important to distinguish those mentioned as background context (no contribution) from those where the aim of the study is positioned in terms of sustainable approaches (contribution).
    For example, an article assessing the people’s preparation and information on climate change could be interpreted as a contribution to SDG 4 (quality education), SDG 12 (responsible production and consumption), and SDG 13 (climate action) which share the theme of sustainable knowledge promotion. However, only the latter would be appropriate in this instance because SDG 4 is about scholastic and academic contexts and SDG 12 is about sustainable lifestyles, consumption and production patterns. Another example is the connection between SDG 5 and SDG 10. Both aim to reduce inequalities. Goal 10 tackles discrimination and inclusion in a broader sense, encompassing economic, social, and political aspects and migratory phenomena. Goal 5 shares the same equality objective but is focused on gender equality and its challenges. Articles tackling these issues from both angles (general and gender-specific) should be tagged with both SDGs; otherwise, the purpose of the paper should determine which SDG is appropriate.

    Is it necessary to read/understand all the areas/aspects of all SDGs?
    Not all the aspects, but when deciding about the contribution of a paper to a given SDG, it is important to be aware of other related SDGs, because this helps understand the scope and boundaries of the SDG under consideration.
    If the annotator is not an SDG expert, they should read the first paragraph of each SDG description card (for example, to know that SDG 2 about hunger, sustainable agriculture, and access to food). When in doubt, they can refer to the detailed description/areas.


    SDG 6: Ensure availability and sustainable management of water and sanitation for all
    Articles contribute to SDG 6 if they are about water, particularly its accessibility, safety, sanitation and management. For example, articles that mention or may have an impact on the following areas should be considered relevant:
    - Access to safe and affordable drinking water
    - Water quality
    - Improve safe sanitation and hygiene
    - Safe wastewater treatment
    - Water recycling and safe reuse
    - Water-use efficiency and management at all levels
    - Sustainable withdrawals and supply of freshwater to address water scarcity
    - Protection and restoration of freshwater-related ecosystems (e.g. oasis, wetlands, rivers, lakes...)
    - People suffering from water scarcity
    - Support of developing countries in water- and sanitation-related activities
    - Support local participation in sanitation management

    Example:
    Articles that study systems to enhance water-use use management in vertical farms or develop technology to improve water sanitization from certain pollutants should be considered relevant to SDG 6.
    Articles that describe the role of properly treated wastewaters in agriculture irrigation or the harmful effects of water pollution, highlighting the urgency of correct sanitation intervention can be considered relevant to SDG 6. They cannot only if they describe hydraulics systems from merely a technical point of view.
  """},


  {"role": "user", "content": content},
  ]

  prompt = pipeline.tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
  )

  terminators = [
      pipeline.tokenizer.eos_token_id,
      pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  outputs = pipeline(
      prompt,
      max_new_tokens=1000,
      eos_token_id=terminators,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
  )

  print(outputs[0]["generated_text"][len(prompt):])
  answers.append(str(outputs[0]["generated_text"][len(prompt):]))

print("Answers:\n")
for a in answers:
  print(a)

#SDG 7
answers = []

for i in range(0, len(df)):
  content = "Title:\n" + df['title'][i] + "\nAbstract:\n" + df['abstract'][i] + "\nKeywords:\n" + df['keywords'][i]

  messages = [
    {"role": "system", "content": """You are an helpful and honest Sustainable Development Goals (SDGs) expert. Your job is to state whether or not the paper contributes to SDG 7: Ensure access to affordable, reliable, sustainable, and modern energy for all. Your answer should be a simple "yes" or "no".

    These are the guidelines:
    How does a paper “contribute to” an SDG?
    A paper can advance sustainable development in many ways: by applying existing technologies, introducing new technologies or methods, examining case studies, or reviewing them to highlight criticalities and possible improvements.
    Importantly, to qualify as a “contribution”, a paper should address an aspect of the SDG's overall theme. These aspects/areas are defined later for each SDG.
    The relation between paper and aspect of the SDG theme could be explicit or implicit: a paper does not necessarily have to explicitly declare the intent of contributing to the SDG (e.g., our work aims at improving sustainable agriculture as described in SDG 2). The relation can also be established based on the paper’s output and the areas of impact. Moreover, the purpose of the study is key to further discriminate the themes the paper contributes to (contribution) from those the paper does not contribute to but mentions them as part of the general background context (no contribution).

    To what extent should the annotator leverage their own knowledge/technical expertise and consider the implicit outcomes or implications of the paper?
    Possible impacts or implications relevant to an SDG should be considered only if they are stated explicitly in the document. For instance, if an article enhances a waste processing system, even if the annotator infers a reduction of energy consumption, the paper should not be tagged with SDG 7 (clean energy) unless the paper makes such a connection explicit.
    Does the paper have to provide a direct/measurable advancement in terms of SDG?

    Not necessarily. Indirect advancements can represent a contribution. These include:
    - surveys on various methods or the current state of the art
    - research producing resources as foundations for future developments
    - enhancements of evaluation methods
    For example, the production of a dataset gathering information about marine pollution to simplify the development of cleaning methods or provide information for sustainable policies and strategies implementation represents a contribution to SDG 14 (life below water).

    How to discriminate between all candidate SDGs that seem related to the paper?
    The SDGs encompass a wide variety of heavily related themes, tackling them from different angles. This might induce the annotator to tag an article with a set of SDGs sharing the same themes or subgoals. However, in choosing the relevant SDGs, it is important to distinguish those mentioned as background context (no contribution) from those where the aim of the study is positioned in terms of sustainable approaches (contribution).
    For example, an article assessing the people’s preparation and information on climate change could be interpreted as a contribution to SDG 4 (quality education), SDG 12 (responsible production and consumption), and SDG 13 (climate action) which share the theme of sustainable knowledge promotion. However, only the latter would be appropriate in this instance because SDG 4 is about scholastic and academic contexts and SDG 12 is about sustainable lifestyles, consumption and production patterns. Another example is the connection between SDG 5 and SDG 10. Both aim to reduce inequalities. Goal 10 tackles discrimination and inclusion in a broader sense, encompassing economic, social, and political aspects and migratory phenomena. Goal 5 shares the same equality objective but is focused on gender equality and its challenges. Articles tackling these issues from both angles (general and gender-specific) should be tagged with both SDGs; otherwise, the purpose of the paper should determine which SDG is appropriate.

    Is it necessary to read/understand all the areas/aspects of all SDGs?
    Not all the aspects, but when deciding about the contribution of a paper to a given SDG, it is important to be aware of other related SDGs, because this helps understand the scope and boundaries of the SDG under consideration.
    If the annotator is not an SDG expert, they should read the first paragraph of each SDG description card (for example, to know that SDG 2 about hunger, sustainable agriculture, and access to food). When in doubt, they can refer to the detailed description/areas.


    SDG 7: Ensure access to affordable, reliable, sustainable, and modern energy for all
    Articles contribute to SDG 7 if they are about energy. For example, articles that mention or may have an impact on the following areas should be considered relevant:
    - Energy efficiency
    - Access to electricity and clean fuels
    - Renewable energy and increase its share in the energy consumption
    - Energy services, infrastructure and technology for developing countries
    - Clean energy research and technology dissemination
    - Infrastructure and technology for supplying modern and sustainable energy services

    Example:
    Articles that assess the effectiveness of applying new energy technologies to farms, improve the building energy efficiency or study methods to enhance access to energy services are relevant to SDG 7.
    Articles analysing the physics of solar panel functioning, without tying it to sustainability concerns, technology enhancement or future developments, are not relevant to SDG 7.
  """},


  {"role": "user", "content": content},
  ]

  prompt = pipeline.tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
  )

  terminators = [
      pipeline.tokenizer.eos_token_id,
      pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  outputs = pipeline(
      prompt,
      max_new_tokens=1000,
      eos_token_id=terminators,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
  )

  print(outputs[0]["generated_text"][len(prompt):])
  answers.append(str(outputs[0]["generated_text"][len(prompt):]))

print("Answers:\n")
for a in answers:
  print(a)

#SDG 8
answers = []

for i in range(0, len(df)):
  content = "Title:\n" + df['title'][i] + "\nAbstract:\n" + df['abstract'][i] + "\nKeywords:\n" + df['keywords'][i]

  messages = [
    {"role": "system", "content": """You are an helpful and honest Sustainable Development Goals (SDGs) expert. Your job is to state whether or not the paper contributes to SDG 8: Promote sustained, inclusive and sustainable economic growth, full and productive employment and decent work for all. Your answer should be a simple "yes" or "no".

    These are the guidelines:
    How does a paper “contribute to” an SDG?
    A paper can advance sustainable development in many ways: by applying existing technologies, introducing new technologies or methods, examining case studies, or reviewing them to highlight criticalities and possible improvements.
    Importantly, to qualify as a “contribution”, a paper should address an aspect of the SDG's overall theme. These aspects/areas are defined later for each SDG.
    The relation between paper and aspect of the SDG theme could be explicit or implicit: a paper does not necessarily have to explicitly declare the intent of contributing to the SDG (e.g., our work aims at improving sustainable agriculture as described in SDG 2). The relation can also be established based on the paper’s output and the areas of impact. Moreover, the purpose of the study is key to further discriminate the themes the paper contributes to (contribution) from those the paper does not contribute to but mentions them as part of the general background context (no contribution).

    To what extent should the annotator leverage their own knowledge/technical expertise and consider the implicit outcomes or implications of the paper?
    Possible impacts or implications relevant to an SDG should be considered only if they are stated explicitly in the document. For instance, if an article enhances a waste processing system, even if the annotator infers a reduction of energy consumption, the paper should not be tagged with SDG 7 (clean energy) unless the paper makes such a connection explicit.
    Does the paper have to provide a direct/measurable advancement in terms of SDG?

    Not necessarily. Indirect advancements can represent a contribution. These include:
    - surveys on various methods or the current state of the art
    - research producing resources as foundations for future developments
    - enhancements of evaluation methods
    For example, the production of a dataset gathering information about marine pollution to simplify the development of cleaning methods or provide information for sustainable policies and strategies implementation represents a contribution to SDG 14 (life below water).

    How to discriminate between all candidate SDGs that seem related to the paper?
    The SDGs encompass a wide variety of heavily related themes, tackling them from different angles. This might induce the annotator to tag an article with a set of SDGs sharing the same themes or subgoals. However, in choosing the relevant SDGs, it is important to distinguish those mentioned as background context (no contribution) from those where the aim of the study is positioned in terms of sustainable approaches (contribution).
    For example, an article assessing the people’s preparation and information on climate change could be interpreted as a contribution to SDG 4 (quality education), SDG 12 (responsible production and consumption), and SDG 13 (climate action) which share the theme of sustainable knowledge promotion. However, only the latter would be appropriate in this instance because SDG 4 is about scholastic and academic contexts and SDG 12 is about sustainable lifestyles, consumption and production patterns. Another example is the connection between SDG 5 and SDG 10. Both aim to reduce inequalities. Goal 10 tackles discrimination and inclusion in a broader sense, encompassing economic, social, and political aspects and migratory phenomena. Goal 5 shares the same equality objective but is focused on gender equality and its challenges. Articles tackling these issues from both angles (general and gender-specific) should be tagged with both SDGs; otherwise, the purpose of the paper should determine which SDG is appropriate.

    Is it necessary to read/understand all the areas/aspects of all SDGs?
    Not all the aspects, but when deciding about the contribution of a paper to a given SDG, it is important to be aware of other related SDGs, because this helps understand the scope and boundaries of the SDG under consideration.
    If the annotator is not an SDG expert, they should read the first paragraph of each SDG description card (for example, to know that SDG 2 about hunger, sustainable agriculture, and access to food). When in doubt, they can refer to the detailed description/areas.


    SDG 8: Promote sustained, inclusive and sustainable economic growth, full and productive employment and decent work for all
    Articles contribute to SDG 8 if they are about economic growth, employment and work conditions. For example, articles that mention or may have an impact on the following areas should be considered relevant:
    - Green economy (decoupling economic growth from environmental degradation) and resource efficiency in consumption and production (material footprint per capita)
    - Sustainable tourism and associated jobs
    - Employment and decent work conditions for all
    - Youth in employment, education and training
    - Child labour and its eradication
    - Enhance economic productivity through diversification, technological upgrading and innovation
    - Employment and decent work with equal pay
    - Compliance of labour rights
    - Safe and secure working environments
    - Occupational injuries
    - Access to banking and financial services

    Example:
    Articles that consider the youths education drop out and analyse the motivations, study methods to reduce work injuries or the effects of tourism on the local population (job and economy-wise) are relevant to SDG 8.
    Articles that report the banking status but not relatively its accessibility improvement, are not relevant to SDG 8.
  """},


  {"role": "user", "content": content},
  ]

  prompt = pipeline.tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
  )

  terminators = [
      pipeline.tokenizer.eos_token_id,
      pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  outputs = pipeline(
      prompt,
      max_new_tokens=1000,
      eos_token_id=terminators,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
  )

  print(outputs[0]["generated_text"][len(prompt):])
  answers.append(str(outputs[0]["generated_text"][len(prompt):]))

print("Answers:\n")
for a in answers:
  print(a)

#SDG 9
answers = []

for i in range(0, len(df)):
  content = "Title:\n" + df['title'][i] + "\nAbstract:\n" + df['abstract'][i] + "\nKeywords:\n" + df['keywords'][i]

  messages = [
    {"role": "system", "content": """You are an helpful and honest Sustainable Development Goals (SDGs) expert. Your job is to state whether or not the paper contributes to SDG 9: Build resilient infrastructure, promote inclusive and sustainable industrialization, and foster innovation. Your answer should be a simple "yes" or "no".

    These are the guidelines:
    How does a paper “contribute to” an SDG?
    A paper can advance sustainable development in many ways: by applying existing technologies, introducing new technologies or methods, examining case studies, or reviewing them to highlight criticalities and possible improvements.
    Importantly, to qualify as a “contribution”, a paper should address an aspect of the SDG's overall theme. These aspects/areas are defined later for each SDG.
    The relation between paper and aspect of the SDG theme could be explicit or implicit: a paper does not necessarily have to explicitly declare the intent of contributing to the SDG (e.g., our work aims at improving sustainable agriculture as described in SDG 2). The relation can also be established based on the paper’s output and the areas of impact. Moreover, the purpose of the study is key to further discriminate the themes the paper contributes to (contribution) from those the paper does not contribute to but mentions them as part of the general background context (no contribution).

    To what extent should the annotator leverage their own knowledge/technical expertise and consider the implicit outcomes or implications of the paper?
    Possible impacts or implications relevant to an SDG should be considered only if they are stated explicitly in the document. For instance, if an article enhances a waste processing system, even if the annotator infers a reduction of energy consumption, the paper should not be tagged with SDG 7 (clean energy) unless the paper makes such a connection explicit.
    Does the paper have to provide a direct/measurable advancement in terms of SDG?

    Not necessarily. Indirect advancements can represent a contribution. These include:
    - surveys on various methods or the current state of the art
    - research producing resources as foundations for future developments
    - enhancements of evaluation methods
    For example, the production of a dataset gathering information about marine pollution to simplify the development of cleaning methods or provide information for sustainable policies and strategies implementation represents a contribution to SDG 14 (life below water).

    How to discriminate between all candidate SDGs that seem related to the paper?
    The SDGs encompass a wide variety of heavily related themes, tackling them from different angles. This might induce the annotator to tag an article with a set of SDGs sharing the same themes or subgoals. However, in choosing the relevant SDGs, it is important to distinguish those mentioned as background context (no contribution) from those where the aim of the study is positioned in terms of sustainable approaches (contribution).
    For example, an article assessing the people’s preparation and information on climate change could be interpreted as a contribution to SDG 4 (quality education), SDG 12 (responsible production and consumption), and SDG 13 (climate action) which share the theme of sustainable knowledge promotion. However, only the latter would be appropriate in this instance because SDG 4 is about scholastic and academic contexts and SDG 12 is about sustainable lifestyles, consumption and production patterns. Another example is the connection between SDG 5 and SDG 10. Both aim to reduce inequalities. Goal 10 tackles discrimination and inclusion in a broader sense, encompassing economic, social, and political aspects and migratory phenomena. Goal 5 shares the same equality objective but is focused on gender equality and its challenges. Articles tackling these issues from both angles (general and gender-specific) should be tagged with both SDGs; otherwise, the purpose of the paper should determine which SDG is appropriate.

    Is it necessary to read/understand all the areas/aspects of all SDGs?
    Not all the aspects, but when deciding about the contribution of a paper to a given SDG, it is important to be aware of other related SDGs, because this helps understand the scope and boundaries of the SDG under consideration.
    If the annotator is not an SDG expert, they should read the first paragraph of each SDG description card (for example, to know that SDG 2 about hunger, sustainable agriculture, and access to food). When in doubt, they can refer to the detailed description/areas.


    SDG 9: Build resilient infrastructure, promote inclusive and sustainable industrialization, and foster innovation
    Articles contribute to SDG 9 if they are about resilient infrastructure and sustainable industry. For example, articles that mention or may have an impact on the following areas should be considered relevant:
    - Industry sustainability and inclusivity
    - Adoption of clean and environmentally sound technologies and industrial processes
    - Infrastructure quality, reliability, sustainability and resiliency
    - Access to infrastructure
    - Access to financial services for small-scale industrial and other enterprises
    - Upgrade all industries and infrastructures for sustainability
    - Scientific research enhancement
    - Upgrade the technological capabilities of industrial sectors
    - Access to information and communications technology

    Example:
    Articles that develop a simulation model forecasting future transportation networks and their performances, gather telecommunication data to optimize the telecommunication network infrastructure or analyse the production process of compound materials and their quality are relevant to SDG 9.
    Articles that explain or suggest projects promoting industrial efficiency or infrastructural solutions without tying them to sustainable development-specific engagement or concerns are not relevant to SDG 9.
  """},


  {"role": "user", "content": content},
  ]

  prompt = pipeline.tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
  )

  terminators = [
      pipeline.tokenizer.eos_token_id,
      pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  outputs = pipeline(
      prompt,
      max_new_tokens=1000,
      eos_token_id=terminators,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
  )

  print(outputs[0]["generated_text"][len(prompt):])
  answers.append(str(outputs[0]["generated_text"][len(prompt):]))

print("Answers:\n")
for a in answers:
  print(a)

#SDG 10
answers = []

for i in range(0, len(df)):
  content = "Title:\n" + df['title'][i] + "\nAbstract:\n" + df['abstract'][i] + "\nKeywords:\n" + df['keywords'][i]

  messages = [
    {"role": "system", "content": """You are an helpful and honest Sustainable Development Goals (SDGs) expert. Your job is to state whether or not the paper contributes to SDG 10: Reduce inequality within and among countries. Your answer should be a simple "yes" or "no".

    These are the guidelines:
    How does a paper “contribute to” an SDG?
    A paper can advance sustainable development in many ways: by applying existing technologies, introducing new technologies or methods, examining case studies, or reviewing them to highlight criticalities and possible improvements.
    Importantly, to qualify as a “contribution”, a paper should address an aspect of the SDG's overall theme. These aspects/areas are defined later for each SDG.
    The relation between paper and aspect of the SDG theme could be explicit or implicit: a paper does not necessarily have to explicitly declare the intent of contributing to the SDG (e.g., our work aims at improving sustainable agriculture as described in SDG 2). The relation can also be established based on the paper’s output and the areas of impact. Moreover, the purpose of the study is key to further discriminate the themes the paper contributes to (contribution) from those the paper does not contribute to but mentions them as part of the general background context (no contribution).

    To what extent should the annotator leverage their own knowledge/technical expertise and consider the implicit outcomes or implications of the paper?
    Possible impacts or implications relevant to an SDG should be considered only if they are stated explicitly in the document. For instance, if an article enhances a waste processing system, even if the annotator infers a reduction of energy consumption, the paper should not be tagged with SDG 7 (clean energy) unless the paper makes such a connection explicit.
    Does the paper have to provide a direct/measurable advancement in terms of SDG?

    Not necessarily. Indirect advancements can represent a contribution. These include:
    - surveys on various methods or the current state of the art
    - research producing resources as foundations for future developments
    - enhancements of evaluation methods
    For example, the production of a dataset gathering information about marine pollution to simplify the development of cleaning methods or provide information for sustainable policies and strategies implementation represents a contribution to SDG 14 (life below water).

    How to discriminate between all candidate SDGs that seem related to the paper?
    The SDGs encompass a wide variety of heavily related themes, tackling them from different angles. This might induce the annotator to tag an article with a set of SDGs sharing the same themes or subgoals. However, in choosing the relevant SDGs, it is important to distinguish those mentioned as background context (no contribution) from those where the aim of the study is positioned in terms of sustainable approaches (contribution).
    For example, an article assessing the people’s preparation and information on climate change could be interpreted as a contribution to SDG 4 (quality education), SDG 12 (responsible production and consumption), and SDG 13 (climate action) which share the theme of sustainable knowledge promotion. However, only the latter would be appropriate in this instance because SDG 4 is about scholastic and academic contexts and SDG 12 is about sustainable lifestyles, consumption and production patterns. Another example is the connection between SDG 5 and SDG 10. Both aim to reduce inequalities. Goal 10 tackles discrimination and inclusion in a broader sense, encompassing economic, social, and political aspects and migratory phenomena. Goal 5 shares the same equality objective but is focused on gender equality and its challenges. Articles tackling these issues from both angles (general and gender-specific) should be tagged with both SDGs; otherwise, the purpose of the paper should determine which SDG is appropriate.

    Is it necessary to read/understand all the areas/aspects of all SDGs?
    Not all the aspects, but when deciding about the contribution of a paper to a given SDG, it is important to be aware of other related SDGs, because this helps understand the scope and boundaries of the SDG under consideration.
    If the annotator is not an SDG expert, they should read the first paragraph of each SDG description card (for example, to know that SDG 2 about hunger, sustainable agriculture, and access to food). When in doubt, they can refer to the detailed description/areas.


    SDG 10: Reduce inequality within and among countries
    Articles contribute to SDG 10 if they are about economic and social inequalities, and discrimination within and among countries. For example, articles that mention or may have an impact on the following areas should be considered relevant:
    - Income growth inequalities
    - Universal social, economic and political inclusion
    - Equal opportunities and inequalities of outcome (reduction of discriminatory practices)
    - Fiscal, wage and social protection policies adoption to achieve greater equality
    - Financial markets regulation
    - Voting rights in developing countries
    - Migration policy planning and management to facilitate orderly, safe, regular and responsible migration
    - Reduction of transaction costs of migrant remittances

    Example:
    Articles that study inequality measures for developing countries, develop new tools which help migrants understanding the host country’s regulations or utilize technology, like Speech to text, improving accessibility and social inclusion of vulnerable subjects (e.g. non-hearing and older people) are relevant to SDG 10.
    Articles that propose methods to improve teaching and gender equity in physical education classes are not relevant to SDG 10 because they do not tackle the issue from an inclusion perspective.
 """},


  {"role": "user", "content": content},
  ]

  prompt = pipeline.tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
  )

  terminators = [
      pipeline.tokenizer.eos_token_id,
      pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  outputs = pipeline(
      prompt,
      max_new_tokens=1000,
      eos_token_id=terminators,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
  )

  print(outputs[0]["generated_text"][len(prompt):])
  answers.append(str(outputs[0]["generated_text"][len(prompt):]))

print("Answers:\n")
for a in answers:
  print(a)

#SDG 11
answers = []

for i in range(0, len(df)):
  content = "Title:\n" + df['title'][i] + "\nAbstract:\n" + df['abstract'][i] + "\nKeywords:\n" + df['keywords'][i]

  messages = [
    {"role": "system", "content": """You are an helpful and honest Sustainable Development Goals (SDGs) expert. Your job is to state whether or not the paper contributes to SDG 11: Make cities and human settlements inclusive, safe, resistent and sustainable. Your answer should be a simple "yes" or "no".

    These are the guidelines:
    How does a paper “contribute to” an SDG?
    A paper can advance sustainable development in many ways: by applying existing technologies, introducing new technologies or methods, examining case studies, or reviewing them to highlight criticalities and possible improvements.
    Importantly, to qualify as a “contribution”, a paper should address an aspect of the SDG's overall theme. These aspects/areas are defined later for each SDG.
    The relation between paper and aspect of the SDG theme could be explicit or implicit: a paper does not necessarily have to explicitly declare the intent of contributing to the SDG (e.g., our work aims at improving sustainable agriculture as described in SDG 2). The relation can also be established based on the paper’s output and the areas of impact. Moreover, the purpose of the study is key to further discriminate the themes the paper contributes to (contribution) from those the paper does not contribute to but mentions them as part of the general background context (no contribution).

    To what extent should the annotator leverage their own knowledge/technical expertise and consider the implicit outcomes or implications of the paper?
    Possible impacts or implications relevant to an SDG should be considered only if they are stated explicitly in the document. For instance, if an article enhances a waste processing system, even if the annotator infers a reduction of energy consumption, the paper should not be tagged with SDG 7 (clean energy) unless the paper makes such a connection explicit.
    Does the paper have to provide a direct/measurable advancement in terms of SDG?

    Not necessarily. Indirect advancements can represent a contribution. These include:
    - surveys on various methods or the current state of the art
    - research producing resources as foundations for future developments
    - enhancements of evaluation methods
    For example, the production of a dataset gathering information about marine pollution to simplify the development of cleaning methods or provide information for sustainable policies and strategies implementation represents a contribution to SDG 14 (life below water).

    How to discriminate between all candidate SDGs that seem related to the paper?
    The SDGs encompass a wide variety of heavily related themes, tackling them from different angles. This might induce the annotator to tag an article with a set of SDGs sharing the same themes or subgoals. However, in choosing the relevant SDGs, it is important to distinguish those mentioned as background context (no contribution) from those where the aim of the study is positioned in terms of sustainable approaches (contribution).
    For example, an article assessing the people’s preparation and information on climate change could be interpreted as a contribution to SDG 4 (quality education), SDG 12 (responsible production and consumption), and SDG 13 (climate action) which share the theme of sustainable knowledge promotion. However, only the latter would be appropriate in this instance because SDG 4 is about scholastic and academic contexts and SDG 12 is about sustainable lifestyles, consumption and production patterns. Another example is the connection between SDG 5 and SDG 10. Both aim to reduce inequalities. Goal 10 tackles discrimination and inclusion in a broader sense, encompassing economic, social, and political aspects and migratory phenomena. Goal 5 shares the same equality objective but is focused on gender equality and its challenges. Articles tackling these issues from both angles (general and gender-specific) should be tagged with both SDGs; otherwise, the purpose of the paper should determine which SDG is appropriate.

    Is it necessary to read/understand all the areas/aspects of all SDGs?
    Not all the aspects, but when deciding about the contribution of a paper to a given SDG, it is important to be aware of other related SDGs, because this helps understand the scope and boundaries of the SDG under consideration.
    If the annotator is not an SDG expert, they should read the first paragraph of each SDG description card (for example, to know that SDG 2 about hunger, sustainable agriculture, and access to food). When in doubt, they can refer to the detailed description/areas.


    SDG 11: Make cities and human settlements inclusive, safe, resistent and sustainable
    Articles contribute to SDG 11 if they are about sustainable and safe cities and settlements. For example, articles that mention or may have an impact on the following areas should be considered relevant:
    - Urban settlements environmental impact (e.g. waste management, air pollution...)
    - Sustainable urbanization, planning and management
    - Protection of cultural heritage, both human and natural
    - Human settlements adopting and implementing policies and plans towards inclusion, resource efficiency, mitigation and adaptation to climate change, resilience to disasters
    - Reduction of losses due to natural disasters such as human lives and economic losses.
    - Disaster risk management policies
    - Sustainable and resilient structures
    - Sustainable, safe, accessible and affordable transportation (especially public transportation)
    - Safe and affordable housing
    - Safe, inclusive and accessible spaces for all

    Example:
    Articles that study the requalification of old buildings to improve energy efficiency and sustainability, requalification of abandoned urban areas with sustainable green spaces, or assess road safety measures should be considered relevant to SDG 11.
    Articles that explain or suggest projects promoting urban solutions or suggesting new structural buildings technologies without tying them to sustainable development-specific engagement or concerns are not relevant to SDG 11.
  """},


  {"role": "user", "content": content},
  ]

  prompt = pipeline.tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
  )

  terminators = [
      pipeline.tokenizer.eos_token_id,
      pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  outputs = pipeline(
      prompt,
      max_new_tokens=1000,
      eos_token_id=terminators,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
  )

  print(outputs[0]["generated_text"][len(prompt):])
  answers.append(str(outputs[0]["generated_text"][len(prompt):]))

print("Answers:\n")
for a in answers:
  print(a)

#SDG 12
answers = []

for i in range(0, len(df)):
  content = "Title:\n" + df['title'][i] + "\nAbstract:\n" + df['abstract'][i] + "\nKeywords:\n" + df['keywords'][i]

  messages = [
    {"role": "system", "content": """You are an helpful and honest Sustainable Development Goals (SDGs) expert. Your job is to state whether or not the paper contributes to SDG 12: Ensure sustainable consumption and production patterns. Your answer should be a simple "yes" or "no".

    These are the guidelines:
    How does a paper “contribute to” an SDG?
    A paper can advance sustainable development in many ways: by applying existing technologies, introducing new technologies or methods, examining case studies, or reviewing them to highlight criticalities and possible improvements.
    Importantly, to qualify as a “contribution”, a paper should address an aspect of the SDG's overall theme. These aspects/areas are defined later for each SDG.
    The relation between paper and aspect of the SDG theme could be explicit or implicit: a paper does not necessarily have to explicitly declare the intent of contributing to the SDG (e.g., our work aims at improving sustainable agriculture as described in SDG 2). The relation can also be established based on the paper’s output and the areas of impact. Moreover, the purpose of the study is key to further discriminate the themes the paper contributes to (contribution) from those the paper does not contribute to but mentions them as part of the general background context (no contribution).

    To what extent should the annotator leverage their own knowledge/technical expertise and consider the implicit outcomes or implications of the paper?
    Possible impacts or implications relevant to an SDG should be considered only if they are stated explicitly in the document. For instance, if an article enhances a waste processing system, even if the annotator infers a reduction of energy consumption, the paper should not be tagged with SDG 7 (clean energy) unless the paper makes such a connection explicit.
    Does the paper have to provide a direct/measurable advancement in terms of SDG?

    Not necessarily. Indirect advancements can represent a contribution. These include:
    - surveys on various methods or the current state of the art
    - research producing resources as foundations for future developments
    - enhancements of evaluation methods
    For example, the production of a dataset gathering information about marine pollution to simplify the development of cleaning methods or provide information for sustainable policies and strategies implementation represents a contribution to SDG 14 (life below water).

    How to discriminate between all candidate SDGs that seem related to the paper?
    The SDGs encompass a wide variety of heavily related themes, tackling them from different angles. This might induce the annotator to tag an article with a set of SDGs sharing the same themes or subgoals. However, in choosing the relevant SDGs, it is important to distinguish those mentioned as background context (no contribution) from those where the aim of the study is positioned in terms of sustainable approaches (contribution).
    For example, an article assessing the people’s preparation and information on climate change could be interpreted as a contribution to SDG 4 (quality education), SDG 12 (responsible production and consumption), and SDG 13 (climate action) which share the theme of sustainable knowledge promotion. However, only the latter would be appropriate in this instance because SDG 4 is about scholastic and academic contexts and SDG 12 is about sustainable lifestyles, consumption and production patterns. Another example is the connection between SDG 5 and SDG 10. Both aim to reduce inequalities. Goal 10 tackles discrimination and inclusion in a broader sense, encompassing economic, social, and political aspects and migratory phenomena. Goal 5 shares the same equality objective but is focused on gender equality and its challenges. Articles tackling these issues from both angles (general and gender-specific) should be tagged with both SDGs; otherwise, the purpose of the paper should determine which SDG is appropriate.

    Is it necessary to read/understand all the areas/aspects of all SDGs?
    Not all the aspects, but when deciding about the contribution of a paper to a given SDG, it is important to be aware of other related SDGs, because this helps understand the scope and boundaries of the SDG under consideration.
    If the annotator is not an SDG expert, they should read the first paragraph of each SDG description card (for example, to know that SDG 2 about hunger, sustainable agriculture, and access to food). When in doubt, they can refer to the detailed description/areas.


    SDG 12: Ensure sustainable consumption and production patterns
    Articles contribute to SDG 12 if they are about waste production and recycling, sustainable practices and lifestyle. For example, articles that mention or may have an impact on the following areas should be considered relevant:
    - Sustainable consumption and production action plans
    - Sustainable use of natural resources
    - Food loss at all levels
    - Management of chemicals and waste
    - Release of chemicals and waste harmful to health and environment
    - Waste generation reduction through prevention, reduction, recycling and reuse
    - Sustainable policies adoption in companies
    - Sustainable public procurement practices promotion
    - Understanding of sustainable lifestyles (e.g. education for sustainable development)
    - Sustainable tourism impacts monitoring
    - Assistance for developing countries' capacity for sustainable production
    - Market distortions that encourage wasteful consumption
    - Fossil-fuel subsidies

    Example:
    Articles that study the integration of wastewater recycling methods in farms, that develop new technologies for removing plastic wastes from the environment or try to quantify the still edible food in wastes are relevant to SDG 12.
    Articles that gather information on waste management new strategies, i.e. just by technical point of view, are not relevant to SDG 12 if they are not related to sustainable consumption or production systems.
  """},


  {"role": "user", "content": content},
  ]

  prompt = pipeline.tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
  )

  terminators = [
      pipeline.tokenizer.eos_token_id,
      pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  outputs = pipeline(
      prompt,
      max_new_tokens=1000,
      eos_token_id=terminators,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
  )

  print(outputs[0]["generated_text"][len(prompt):])
  answers.append(str(outputs[0]["generated_text"][len(prompt):]))

print("Answers:\n")
for a in answers:
  print(a)

#SDG 13
answers = []

for i in range(0, len(df)):
  content = "Title:\n" + df['title'][i] + "\nAbstract:\n" + df['abstract'][i] + "\nKeywords:\n" + df['keywords'][i]

  messages = [
    {"role": "system", "content": """You are an helpful and honest Sustainable Development Goals (SDGs) expert. Your job is to state whether or not the paper contributes to SDG 13: Take urgent action to combat climate change and its impacts. Your answer should be a simple "yes" or "no".

    These are the guidelines:
    How does a paper “contribute to” an SDG?
    A paper can advance sustainable development in many ways: by applying existing technologies, introducing new technologies or methods, examining case studies, or reviewing them to highlight criticalities and possible improvements.
    Importantly, to qualify as a “contribution”, a paper should address an aspect of the SDG's overall theme. These aspects/areas are defined later for each SDG.
    The relation between paper and aspect of the SDG theme could be explicit or implicit: a paper does not necessarily have to explicitly declare the intent of contributing to the SDG (e.g., our work aims at improving sustainable agriculture as described in SDG 2). The relation can also be established based on the paper’s output and the areas of impact. Moreover, the purpose of the study is key to further discriminate the themes the paper contributes to (contribution) from those the paper does not contribute to but mentions them as part of the general background context (no contribution).

    To what extent should the annotator leverage their own knowledge/technical expertise and consider the implicit outcomes or implications of the paper?
    Possible impacts or implications relevant to an SDG should be considered only if they are stated explicitly in the document. For instance, if an article enhances a waste processing system, even if the annotator infers a reduction of energy consumption, the paper should not be tagged with SDG 7 (clean energy) unless the paper makes such a connection explicit.
    Does the paper have to provide a direct/measurable advancement in terms of SDG?

    Not necessarily. Indirect advancements can represent a contribution. These include:
    - surveys on various methods or the current state of the art
    - research producing resources as foundations for future developments
    - enhancements of evaluation methods
    For example, the production of a dataset gathering information about marine pollution to simplify the development of cleaning methods or provide information for sustainable policies and strategies implementation represents a contribution to SDG 14 (life below water).

    How to discriminate between all candidate SDGs that seem related to the paper?
    The SDGs encompass a wide variety of heavily related themes, tackling them from different angles. This might induce the annotator to tag an article with a set of SDGs sharing the same themes or subgoals. However, in choosing the relevant SDGs, it is important to distinguish those mentioned as background context (no contribution) from those where the aim of the study is positioned in terms of sustainable approaches (contribution).
    For example, an article assessing the people’s preparation and information on climate change could be interpreted as a contribution to SDG 4 (quality education), SDG 12 (responsible production and consumption), and SDG 13 (climate action) which share the theme of sustainable knowledge promotion. However, only the latter would be appropriate in this instance because SDG 4 is about scholastic and academic contexts and SDG 12 is about sustainable lifestyles, consumption and production patterns. Another example is the connection between SDG 5 and SDG 10. Both aim to reduce inequalities. Goal 10 tackles discrimination and inclusion in a broader sense, encompassing economic, social, and political aspects and migratory phenomena. Goal 5 shares the same equality objective but is focused on gender equality and its challenges. Articles tackling these issues from both angles (general and gender-specific) should be tagged with both SDGs; otherwise, the purpose of the paper should determine which SDG is appropriate.

    Is it necessary to read/understand all the areas/aspects of all SDGs?
    Not all the aspects, but when deciding about the contribution of a paper to a given SDG, it is important to be aware of other related SDGs, because this helps understand the scope and boundaries of the SDG under consideration.
    If the annotator is not an SDG expert, they should read the first paragraph of each SDG description card (for example, to know that SDG 2 about hunger, sustainable agriculture, and access to food). When in doubt, they can refer to the detailed description/areas.


    SDG 13: Take urgent action to combat climate change and its impacts
    Articles contribute to SDG 13 if they are about climate change and taking action to mitigate its effects. For example, articles that mention or may have an impact on the following areas should be considered relevant:
    - Information and indicators of climate change
    - Education and awareness on climate change
    - Climate change mitigation, adaptation, impact reduction and early warning
    - Greenhouse gas emissions
    - Integration of climate change measures in policies
    - Local and national disaster risk management
    - Support for planning and management in least-developed countries

    Example:
    Articles that improve technology to reduce Greenhouse gas emissions, assess the population knowledge on climate change or develop system to detect or predict natural disasters are relevant to SDG 13.
    Articles that gather information on some technical solutions in CO2 storage strategies or emissions control systems are not relevant to SDG 13 if the information is not related to sustainable climate change actions.
  """},


  {"role": "user", "content": content},
  ]

  prompt = pipeline.tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
  )

  terminators = [
      pipeline.tokenizer.eos_token_id,
      pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  outputs = pipeline(
      prompt,
      max_new_tokens=1000,
      eos_token_id=terminators,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
  )

  print(outputs[0]["generated_text"][len(prompt):])
  answers.append(str(outputs[0]["generated_text"][len(prompt):]))

print("Answers:\n")
for a in answers:
  print(a)

#SDG 14
answers = []

for i in range(0, len(df)):
  content = "Title:\n" + df['title'][i] + "\nAbstract:\n" + df['abstract'][i] + "\nKeywords:\n" + df['keywords'][i]

  messages = [
    {"role": "system", "content": """You are an helpful and honest Sustainable Development Goals (SDGs) expert. Your job is to state whether or not the paper contributes to SDG 14: Conserve and sustainably use the oceans, seas and marine resources for sustainable development. Your answer should be a simple "yes" or "no".

    These are the guidelines:
    How does a paper “contribute to” an SDG?
    A paper can advance sustainable development in many ways: by applying existing technologies, introducing new technologies or methods, examining case studies, or reviewing them to highlight criticalities and possible improvements.
    Importantly, to qualify as a “contribution”, a paper should address an aspect of the SDG's overall theme. These aspects/areas are defined later for each SDG.
    The relation between paper and aspect of the SDG theme could be explicit or implicit: a paper does not necessarily have to explicitly declare the intent of contributing to the SDG (e.g., our work aims at improving sustainable agriculture as described in SDG 2). The relation can also be established based on the paper’s output and the areas of impact. Moreover, the purpose of the study is key to further discriminate the themes the paper contributes to (contribution) from those the paper does not contribute to but mentions them as part of the general background context (no contribution).

    To what extent should the annotator leverage their own knowledge/technical expertise and consider the implicit outcomes or implications of the paper?
    Possible impacts or implications relevant to an SDG should be considered only if they are stated explicitly in the document. For instance, if an article enhances a waste processing system, even if the annotator infers a reduction of energy consumption, the paper should not be tagged with SDG 7 (clean energy) unless the paper makes such a connection explicit.
    Does the paper have to provide a direct/measurable advancement in terms of SDG?

    Not necessarily. Indirect advancements can represent a contribution. These include:
    - surveys on various methods or the current state of the art
    - research producing resources as foundations for future developments
    - enhancements of evaluation methods
    For example, the production of a dataset gathering information about marine pollution to simplify the development of cleaning methods or provide information for sustainable policies and strategies implementation represents a contribution to SDG 14 (life below water).

    How to discriminate between all candidate SDGs that seem related to the paper?
    The SDGs encompass a wide variety of heavily related themes, tackling them from different angles. This might induce the annotator to tag an article with a set of SDGs sharing the same themes or subgoals. However, in choosing the relevant SDGs, it is important to distinguish those mentioned as background context (no contribution) from those where the aim of the study is positioned in terms of sustainable approaches (contribution).
    For example, an article assessing the people’s preparation and information on climate change could be interpreted as a contribution to SDG 4 (quality education), SDG 12 (responsible production and consumption), and SDG 13 (climate action) which share the theme of sustainable knowledge promotion. However, only the latter would be appropriate in this instance because SDG 4 is about scholastic and academic contexts and SDG 12 is about sustainable lifestyles, consumption and production patterns. Another example is the connection between SDG 5 and SDG 10. Both aim to reduce inequalities. Goal 10 tackles discrimination and inclusion in a broader sense, encompassing economic, social, and political aspects and migratory phenomena. Goal 5 shares the same equality objective but is focused on gender equality and its challenges. Articles tackling these issues from both angles (general and gender-specific) should be tagged with both SDGs; otherwise, the purpose of the paper should determine which SDG is appropriate.

    Is it necessary to read/understand all the areas/aspects of all SDGs?
    Not all the aspects, but when deciding about the contribution of a paper to a given SDG, it is important to be aware of other related SDGs, because this helps understand the scope and boundaries of the SDG under consideration.
    If the annotator is not an SDG expert, they should read the first paragraph of each SDG description card (for example, to know that SDG 2 about hunger, sustainable agriculture, and access to food). When in doubt, they can refer to the detailed description/areas.


    SDG 14: Conserve and sustainably use the oceans, seas and marine resources for sustainable development
    Articles contribute to SDG 14 if they are about protection and restoration of marine ecosystems. For example, articles that mention or may have an impact on the following areas should be considered relevant:
    - Protection and restoration of marine ecosystems
    - Conservation and sustainable use of oceans and all their resources
    - Conservation of coastal areas
    - Marine pollution
    - Ocean acidification and its impacts
    - Sustainable fishing
    - Fishing regulation both in markets and at sea
    - Promotion and integration of scientific knowledge for marine technology

    Example:
    Articles that develop sustainable fishing plans, gather data on marine phenomena to enhance policy implementation, or a review gathering information on the marine beach littering and highlighting the current gaps are relevant to SDG 14.
    Articles reporting biological studies of marine ecosystems without expressing a relation to possible harmful effects to them, like the impact of pollution or the acidification, are not relevant to SDG 14.
  """},


  {"role": "user", "content": content},
  ]

  prompt = pipeline.tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
  )

  terminators = [
      pipeline.tokenizer.eos_token_id,
      pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  outputs = pipeline(
      prompt,
      max_new_tokens=1000,
      eos_token_id=terminators,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
  )

  print(outputs[0]["generated_text"][len(prompt):])
  answers.append(str(outputs[0]["generated_text"][len(prompt):]))

print("Answers:\n")
for a in answers:
  print(a)

#SDG 15
answers = []

for i in range(0, len(df)):
  content = "Title:\n" + df['title'][i] + "\nAbstract:\n" + df['abstract'][i] + "\nKeywords:\n" + df['keywords'][i]

  messages = [
    {"role": "system", "content": """You are an helpful and honest Sustainable Development Goals (SDGs) expert. Your job is to state whether or not the paper contributes to SDG 15: Protect, restore and promote sustainable use of terrestrial ecosystems, sustainably manage forests, combat desertification, and halt reverse land degradation and halt biodiversity loss. Your answer should be a simple "yes" or "no".

    These are the guidelines:
    How does a paper “contribute to” an SDG?
    A paper can advance sustainable development in many ways: by applying existing technologies, introducing new technologies or methods, examining case studies, or reviewing them to highlight criticalities and possible improvements.
    Importantly, to qualify as a “contribution”, a paper should address an aspect of the SDG's overall theme. These aspects/areas are defined later for each SDG.
    The relation between paper and aspect of the SDG theme could be explicit or implicit: a paper does not necessarily have to explicitly declare the intent of contributing to the SDG (e.g., our work aims at improving sustainable agriculture as described in SDG 2). The relation can also be established based on the paper’s output and the areas of impact. Moreover, the purpose of the study is key to further discriminate the themes the paper contributes to (contribution) from those the paper does not contribute to but mentions them as part of the general background context (no contribution).

    To what extent should the annotator leverage their own knowledge/technical expertise and consider the implicit outcomes or implications of the paper?
    Possible impacts or implications relevant to an SDG should be considered only if they are stated explicitly in the document. For instance, if an article enhances a waste processing system, even if the annotator infers a reduction of energy consumption, the paper should not be tagged with SDG 7 (clean energy) unless the paper makes such a connection explicit.
    Does the paper have to provide a direct/measurable advancement in terms of SDG?

    Not necessarily. Indirect advancements can represent a contribution. These include:
    - surveys on various methods or the current state of the art
    - research producing resources as foundations for future developments
    - enhancements of evaluation methods
    For example, the production of a dataset gathering information about marine pollution to simplify the development of cleaning methods or provide information for sustainable policies and strategies implementation represents a contribution to SDG 14 (life below water).

    How to discriminate between all candidate SDGs that seem related to the paper?
    The SDGs encompass a wide variety of heavily related themes, tackling them from different angles. This might induce the annotator to tag an article with a set of SDGs sharing the same themes or subgoals. However, in choosing the relevant SDGs, it is important to distinguish those mentioned as background context (no contribution) from those where the aim of the study is positioned in terms of sustainable approaches (contribution).
    For example, an article assessing the people’s preparation and information on climate change could be interpreted as a contribution to SDG 4 (quality education), SDG 12 (responsible production and consumption), and SDG 13 (climate action) which share the theme of sustainable knowledge promotion. However, only the latter would be appropriate in this instance because SDG 4 is about scholastic and academic contexts and SDG 12 is about sustainable lifestyles, consumption and production patterns. Another example is the connection between SDG 5 and SDG 10. Both aim to reduce inequalities. Goal 10 tackles discrimination and inclusion in a broader sense, encompassing economic, social, and political aspects and migratory phenomena. Goal 5 shares the same equality objective but is focused on gender equality and its challenges. Articles tackling these issues from both angles (general and gender-specific) should be tagged with both SDGs; otherwise, the purpose of the paper should determine which SDG is appropriate.

    Is it necessary to read/understand all the areas/aspects of all SDGs?
    Not all the aspects, but when deciding about the contribution of a paper to a given SDG, it is important to be aware of other related SDGs, because this helps understand the scope and boundaries of the SDG under consideration.
    If the annotator is not an SDG expert, they should read the first paragraph of each SDG description card (for example, to know that SDG 2 about hunger, sustainable agriculture, and access to food). When in doubt, they can refer to the detailed description/areas.


    SDG 15: Protect, restore and promote sustainable use of terrestrial ecosystems, sustainably manage forests, combat desertification, and halt reverse land degradation and halt biodiversity loss
    Articles contribute to SDG 15 if they are about protection of land ecosystem, land degradation. For example, articles that mention or may have an impact on the following areas should be considered relevant:
    - Terrestrial and inland freshwater ecosystems (e.g. forests, wetlands, mountains, drylands...) and their conservation, restoration and sustainable use
    - Forests conservation
    - Forests sustainable management
    - Desertification
    - Land and soil degradation
    - Degradation of natural habitats
    - Loss of biodiversity
    - Regulated sharing of genetic resources
    - Wildlife poaching and trafficking
    - Impact of alien species on land and water ecosystems
    - Integration of ecosystem and biodiversity in planning

    Example:
    Articles that study cultural forest significance aiming at promoting sustainable management, proposing collaborative frameworks for oasis conservation or studying sustainable fertilizers to improve soil quality are relevant to SDG 15.
    Articles that study cultural forest significance, study the characteristics of an ecosystem from a biological point of view without tying it sustainable development concerns or gathering information to tackle these issues in the future are not relevant to SDG 15.
  """},


  {"role": "user", "content": content},
  ]

  prompt = pipeline.tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
  )

  terminators = [
      pipeline.tokenizer.eos_token_id,
      pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  outputs = pipeline(
      prompt,
      max_new_tokens=1000,
      eos_token_id=terminators,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
  )

  print(outputs[0]["generated_text"][len(prompt):])
  answers.append(str(outputs[0]["generated_text"][len(prompt):]))

print("Answers:\n")
for a in answers:
  print(a)

#SDG 16
answers = []

for i in range(0, len(df)):
  content = "Title:\n" + df['title'][i] + "\nAbstract:\n" + df['abstract'][i] + "\nKeywords:\n" + df['keywords'][i]

  messages = [
    {"role": "system", "content": """You are an helpful and honest Sustainable Development Goals (SDGs) expert. Your job is to state whether or not the paper contributes to SDG 16: Promote peaceful and inclusive societies for sustainable development, provide access to justice for all and build effective, accountable, and inclusive institutions at all levels. Your answer should be a simple "yes" or "no".

    These are the guidelines:
    How does a paper “contribute to” an SDG?
    A paper can advance sustainable development in many ways: by applying existing technologies, introducing new technologies or methods, examining case studies, or reviewing them to highlight criticalities and possible improvements.
    Importantly, to qualify as a “contribution”, a paper should address an aspect of the SDG's overall theme. These aspects/areas are defined later for each SDG.
    The relation between paper and aspect of the SDG theme could be explicit or implicit: a paper does not necessarily have to explicitly declare the intent of contributing to the SDG (e.g., our work aims at improving sustainable agriculture as described in SDG 2). The relation can also be established based on the paper’s output and the areas of impact. Moreover, the purpose of the study is key to further discriminate the themes the paper contributes to (contribution) from those the paper does not contribute to but mentions them as part of the general background context (no contribution).

    To what extent should the annotator leverage their own knowledge/technical expertise and consider the implicit outcomes or implications of the paper?
    Possible impacts or implications relevant to an SDG should be considered only if they are stated explicitly in the document. For instance, if an article enhances a waste processing system, even if the annotator infers a reduction of energy consumption, the paper should not be tagged with SDG 7 (clean energy) unless the paper makes such a connection explicit.
    Does the paper have to provide a direct/measurable advancement in terms of SDG?

    Not necessarily. Indirect advancements can represent a contribution. These include:
    - surveys on various methods or the current state of the art
    - research producing resources as foundations for future developments
    - enhancements of evaluation methods
    For example, the production of a dataset gathering information about marine pollution to simplify the development of cleaning methods or provide information for sustainable policies and strategies implementation represents a contribution to SDG 14 (life below water).

    How to discriminate between all candidate SDGs that seem related to the paper?
    The SDGs encompass a wide variety of heavily related themes, tackling them from different angles. This might induce the annotator to tag an article with a set of SDGs sharing the same themes or subgoals. However, in choosing the relevant SDGs, it is important to distinguish those mentioned as background context (no contribution) from those where the aim of the study is positioned in terms of sustainable approaches (contribution).
    For example, an article assessing the people’s preparation and information on climate change could be interpreted as a contribution to SDG 4 (quality education), SDG 12 (responsible production and consumption), and SDG 13 (climate action) which share the theme of sustainable knowledge promotion. However, only the latter would be appropriate in this instance because SDG 4 is about scholastic and academic contexts and SDG 12 is about sustainable lifestyles, consumption and production patterns. Another example is the connection between SDG 5 and SDG 10. Both aim to reduce inequalities. Goal 10 tackles discrimination and inclusion in a broader sense, encompassing economic, social, and political aspects and migratory phenomena. Goal 5 shares the same equality objective but is focused on gender equality and its challenges. Articles tackling these issues from both angles (general and gender-specific) should be tagged with both SDGs; otherwise, the purpose of the paper should determine which SDG is appropriate.

    Is it necessary to read/understand all the areas/aspects of all SDGs?
    Not all the aspects, but when deciding about the contribution of a paper to a given SDG, it is important to be aware of other related SDGs, because this helps understand the scope and boundaries of the SDG under consideration.
    If the annotator is not an SDG expert, they should read the first paragraph of each SDG description card (for example, to know that SDG 2 about hunger, sustainable agriculture, and access to food). When in doubt, they can refer to the detailed description/areas.


    SDG 16: Promote peaceful and inclusive societies for sustainable development, provide access to justice for all and build effective, accountable, and inclusive institutions at all levels
    Articles contribute to SDG 16 if they are about peaceful and safe societies, human rights and institutions. For example, articles that mention or may have an impact on the following areas should be considered relevant:
    - Violence and related deaths
    - Human trafficking
    - Sexual violence
    - Equal access to justice
    - Illicit financial and arms flows
    - Fight against organized crime
    - Corruption and bribery
    - Effective, accountable and transparent institutions at all levels
    - Representation in public institutions and inclusive decision-making at all levels for all
    - Improving participation of developing countries in international organization
    - Legal identity provision (e.g. birth registration)
    - Public access to information
    - Fundamental freedoms protection
    - Non-discriminatory laws and policies for sustainable development

    Example:
    Articles that develop systems improving an institution’s reports, enhancing its accessibility and transparency to the public, or articles proposing measures to reduce bullying or other forms of violence in schools are relevant to SDG 16.
    Articles that explain a process in public institutions management without tying it to inclusive engagement are not relevant to SDG 16.
  """},


  {"role": "user", "content": content},
  ]

  prompt = pipeline.tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
  )

  terminators = [
      pipeline.tokenizer.eos_token_id,
      pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  outputs = pipeline(
      prompt,
      max_new_tokens=1000,
      eos_token_id=terminators,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
  )

  print(outputs[0]["generated_text"][len(prompt):])
  answers.append(str(outputs[0]["generated_text"][len(prompt):]))

print("Answers:\n")
for a in answers:
  print(a)

#SDG 17
answers = []

for i in range(0, len(df)):
  content = "Title:\n" + df['title'][i] + "\nAbstract:\n" + df['abstract'][i] + "\nKeywords:\n" + df['keywords'][i]

  messages = [
    {"role": "system", "content": """You are an helpful and honest Sustainable Development Goals (SDGs) expert. Your job is to state whether or not the paper contributes to SDG 17: Strenghten the means of implementation and revitalize the Global Partnership for Sustainable Development. Your answer should be a simple "yes" or "no".

    These are the guidelines:
    How does a paper “contribute to” an SDG?
    A paper can advance sustainable development in many ways: by applying existing technologies, introducing new technologies or methods, examining case studies, or reviewing them to highlight criticalities and possible improvements.
    Importantly, to qualify as a “contribution”, a paper should address an aspect of the SDG's overall theme. These aspects/areas are defined later for each SDG.
    The relation between paper and aspect of the SDG theme could be explicit or implicit: a paper does not necessarily have to explicitly declare the intent of contributing to the SDG (e.g., our work aims at improving sustainable agriculture as described in SDG 2). The relation can also be established based on the paper’s output and the areas of impact. Moreover, the purpose of the study is key to further discriminate the themes the paper contributes to (contribution) from those the paper does not contribute to but mentions them as part of the general background context (no contribution).

    To what extent should the annotator leverage their own knowledge/technical expertise and consider the implicit outcomes or implications of the paper?
    Possible impacts or implications relevant to an SDG should be considered only if they are stated explicitly in the document. For instance, if an article enhances a waste processing system, even if the annotator infers a reduction of energy consumption, the paper should not be tagged with SDG 7 (clean energy) unless the paper makes such a connection explicit.
    Does the paper have to provide a direct/measurable advancement in terms of SDG?

    Not necessarily. Indirect advancements can represent a contribution. These include:
    - surveys on various methods or the current state of the art
    - research producing resources as foundations for future developments
    - enhancements of evaluation methods
    For example, the production of a dataset gathering information about marine pollution to simplify the development of cleaning methods or provide information for sustainable policies and strategies implementation represents a contribution to SDG 14 (life below water).

    How to discriminate between all candidate SDGs that seem related to the paper?
    The SDGs encompass a wide variety of heavily related themes, tackling them from different angles. This might induce the annotator to tag an article with a set of SDGs sharing the same themes or subgoals. However, in choosing the relevant SDGs, it is important to distinguish those mentioned as background context (no contribution) from those where the aim of the study is positioned in terms of sustainable approaches (contribution).
    For example, an article assessing the people’s preparation and information on climate change could be interpreted as a contribution to SDG 4 (quality education), SDG 12 (responsible production and consumption), and SDG 13 (climate action) which share the theme of sustainable knowledge promotion. However, only the latter would be appropriate in this instance because SDG 4 is about scholastic and academic contexts and SDG 12 is about sustainable lifestyles, consumption and production patterns. Another example is the connection between SDG 5 and SDG 10. Both aim to reduce inequalities. Goal 10 tackles discrimination and inclusion in a broader sense, encompassing economic, social, and political aspects and migratory phenomena. Goal 5 shares the same equality objective but is focused on gender equality and its challenges. Articles tackling these issues from both angles (general and gender-specific) should be tagged with both SDGs; otherwise, the purpose of the paper should determine which SDG is appropriate.

    Is it necessary to read/understand all the areas/aspects of all SDGs?
    Not all the aspects, but when deciding about the contribution of a paper to a given SDG, it is important to be aware of other related SDGs, because this helps understand the scope and boundaries of the SDG under consideration.
    If the annotator is not an SDG expert, they should read the first paragraph of each SDG description card (for example, to know that SDG 2 about hunger, sustainable agriculture, and access to food). When in doubt, they can refer to the detailed description/areas.


    SDG 17: Strenghten the means of implementation and revitalize the Global Partnership for Sustainable Development
    Articles contribute to SDG 17 if they are about global partnership and cooperation for sustainable development and its means of implementation. For example, articles that mention or may have an impact on the following areas should be considered relevant:
    - Global partnership for sustainable development
    - Measuring progress in sustainable development
    - Improving the capacity of developing countries for tax and other revenue collection
    - Financial support for developing countries
    - Development assistance to least-developed countries
    - Assistance to developing countries in debt sustainability
    - Sustainable technology sharing with developing countries
    - Universal, equal and regulated trading system
    - Improving the exports of developing countries
    - Improving the scientific capacity of developing countries for sustainable development
    - Scientific and technological cooperation
    - Policy coherence for sustainable development
    - Sustainable technologies in developing countries
    - Sovereignty in the implementation of policies for poverty eradication and SDGs

    Example:
    Articles that propose a collaboration framework to protect or maintain an ecosystem are relevant to SDG 17.
    Articles that explain a process/project promoted by local policy makers or institutions without tying it to sustainable development-specific engagement or concerns are not relevant to SDG 17.
  """},


  {"role": "user", "content": content},
  ]

  prompt = pipeline.tokenizer.apply_chat_template(
          messages,
          tokenize=False,
          add_generation_prompt=True
  )

  terminators = [
      pipeline.tokenizer.eos_token_id,
      pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
  ]

  outputs = pipeline(
      prompt,
      max_new_tokens=1000,
      eos_token_id=terminators,
      do_sample=True,
      temperature=0.6,
      top_p=0.9,
  )

  print(outputs[0]["generated_text"][len(prompt):])
  answers.append(str(outputs[0]["generated_text"][len(prompt):]))

print("Answers:\n")
for a in answers:
  print(a)