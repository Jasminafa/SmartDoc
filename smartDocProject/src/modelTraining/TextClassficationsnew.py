

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
nltk.download("punkt", quiet=True)
metric = evaluate.load("rouge")



MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)


train_path = "C:/Users/Shaha/OneDrive/Desktop/train1.xlsx"
train_df = pd.read_excel(train_path)
train_dts = Dataset.from_pandas(train_df)
full_dataset = train_dts.train_test_split(test_size=0.2)
print(full_dataset)



int_to_str = {0:'0', 1:'1',2:'2', 3:'3'}

# Define the preprocessing function

def preprocess_function(examples):
   """ tokenize the text, and set the labels"""
   # The "inputs" are the tokenized answer:
   inputs = examples["text"]
   model_inputs = tokenizer(inputs, max_length=200, padding=True, truncation=True)
    
   labels = tokenizer(text_target=int_to_str[examples["target"]], padding=True, truncation=True)
   model_inputs["labels"] = labels["input_ids"]

   return model_inputs

tokenized_dataset = full_dataset.map(preprocess_function, batched=False, remove_columns=['id','text','target'])




def compute_metrics(eval_preds):
   preds, labels = eval_preds

   # decode preds and labels
   labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
   decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
   decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

   # rougeLSum expects newline after each sentence
   decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
   decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

   result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
  
   return result


# Global Parameters
L_RATE = 3e-4
BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH = 4
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIM = 3
NUM_EPOCHS = 2

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
   output_dir="D:/results",
   eval_strategy="epoch",
   learning_rate=L_RATE,
   per_device_train_batch_size=BATCH_SIZE,
   per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
   weight_decay=WEIGHT_DECAY,
   save_total_limit=SAVE_TOTAL_LIM,
   num_train_epochs=NUM_EPOCHS,
   predict_with_generate=True,
   push_to_hub=False
)

trainer = Seq2SeqTrainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_dataset["train"],
   eval_dataset=tokenized_dataset["test"],
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics
)


trainer.train()


MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)




test_path = 'C:/Users/Shaha/OneDrive/Desktop/test.xlsx'
test_df = pd.read_excel(test_path)
test_dts = Dataset.from_pandas(test_df)
print(test_df.head())



last_checkpoint = "D:/results/checkpoint-20"
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









