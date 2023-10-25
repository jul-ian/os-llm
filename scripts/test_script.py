from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig



"""
model_name = 'google/flan-t5-base'
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
"""

from transformers import GPT2LMHeadModel
tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

question_format = """
1="SIDEWALKS"
2="HOME - OUTDOOR"
3="HOME - INDOOR"
4="SHOPPING MALLS OR STORES"
5="WORKPLACE - OUTDOOR"
6="WORKPLACE - INDOOR"
7="PARK, BEACH, PLAYGROUND, OUTDOOR RECREATION"
8="DRIVING"
9="COMMON AREA OF AN APARTMENT COMPLEX, CONDO"
91="OTHER"
"""
question_format = question_format
shot1 = "ANOTHER PERSONS HOME OUTSIDE=2"

responses = ['A CASINO', 'BASEBALL FIELD', 'BROTHERS HOME INDOORS', 'COLLEGE CAMPUS'] 

upcoding = [4, 7, 3, 5]

completions = []
for response in responses:
    input_string = tokenizer(f'{question_format}\n{shot1}\n{response}=#',
                              return_tensors='pt')
    completion = tokenizer.decode(
        model.generate(
            input_string["input_ids"], 
            max_new_tokens=50,
        )[0], 
        skip_special_tokens=True
    )
    completions.append(completion)
    print('-'*10)
    print(f'{question_format}\n{shot1}\n{response}={completion}')
    print('-'*10)
