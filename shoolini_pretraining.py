
!pip install datasets transformers tokenizer accelerate

from datasets import load_dataset
import os

data=load_dataset('text',data_files='/content/shoolini.txt')

data

from transformers import (
    GPT2TokenizerFast,
    GPT2Config,
    GPT2LMHeadModel,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments
)

tokenizer=GPT2TokenizerFast.from_pretrained("gpt2")

tokenizer.pad_token=tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"])

tokenized_data=data.map(tokenize_function,batched=True,num_proc=4,remove_columns=["text"])

tokenized_data

config=GPT2Config(
    vocab_size=tokenizer.vocab_size,
    bos_token_id=tokenizer.bos_token_id,
    eos_token_id=tokenizer.eos_token_id,
    n_layer=6,
    n_head=6,
    n_embd=384
)

new_model=GPT2LMHeadModel(config)

data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)

training_args=TrainingArguments(
    output_dir="./gpt2-shoolini",
    num_train_epochs=5,
    per_device_train_batch_size=32,
    eval_steps=50,
    save_total_limit=2
)

trainer=Trainer(
    model=new_model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_data["train"]
)

trainer.train()

new_model.save_pretrained("./gpt2-shoolini")

prompt="Where is shoolini University?"

tokenizer.save_pretrained("./gpt2-shoolini")

from transformers import pipeline


text_generator = pipeline("text-generation", model="./gpt2-shoolini")

output=text_generator(prompt, max_length=50, do_sample=True)

print(output[0]['generated_text'])

