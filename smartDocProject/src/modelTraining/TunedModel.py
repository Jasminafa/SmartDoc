

import nltk
nltk.download('punkt') 
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import evaluate
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from tqdm import tqdm



MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)




test_path = 'C:/Users/Shaha/OneDrive/Desktop/test.xlsx'
test_df = pd.read_excel(test_path)
test_dts = Dataset.from_pandas(test_df)
print(test_df.head())



last_checkpoint = "C:/Users/Shaha/OneDrive/Desktop/results/checkpoint-348"
finetuned_model = T5ForConditionalGeneration.from_pretrained(last_checkpoint)
finetuned_tokenizer = AutoTokenizer.from_pretrained(last_checkpoint)




str_to_int = {'0':0, '1':1,'2':2, '3':3}

data = []

for i in tqdm(range(len(test_df))):
   id = test_df['id'][i]
   inputs = test_df['text'][i]
   model_inputs = finetuned_tokenizer(inputs, return_tensors="pt") 
   outputs = finetuned_model.generate(**model_inputs)
   answer = finetuned_tokenizer.decode(outputs[0])
   print(answer)
   data.append({'id': id, 'target': str_to_int[answer[6]]})




df = pd.DataFrame(data)
df.to_excel('C:/Users/Shaha/OneDrive/Desktop/results/submission.xlsx') 


print(df.head())









