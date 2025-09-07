import os
import sys
import xml.etree.ElementTree as ET
import re
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

def get_file(filename_prompt):
    is_in_colab = 'google.colab' in sys.modules
    if not os.path.exists(filename_prompt):
        if is_in_colab:
            try:
                from google.colab import files
                print(f"\nPlease upload the required dataset file: '{filename_prompt}'")
                uploaded = files.upload()
                if not uploaded:
                    print(f"\nERROR: File upload for '{filename_prompt}' was cancelled or failed.")
                    sys.exit()
                actual_filename = list(uploaded.keys())[0]
                os.rename(actual_filename, filename_prompt)
                print(f"Successfully received and saved '{filename_prompt}'.")
                return filename_prompt
            except ImportError:
                pass
        else:
            print("="*80)
            print(f"ERROR: Dataset file '{filename_prompt}' not found.")
            print("="*80)
            sys.exit()
    print(f"Successfully found '{filename_prompt}'.")
    return filename_prompt

def prepare_gpt2_data(corpus_xml_file, output_dir):
    print(f"--- Preparing data for GPT-2 from {corpus_xml_file} ---")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    all_texts = []
    try:
        tree = ET.parse(corpus_xml_file)
        root = tree.getroot()
        for conversation in root.findall('conversation'):
            convo_text = []
            for message in conversation.findall('message'):
                text_content_element = message.find('text')
                if text_content_element is not None and text_content_element.text:
                    clean_text = re.sub(r'<.*?>', '', text_content_element.text).strip()
                    if clean_text:
                        convo_text.append(clean_text)
            if convo_text:
                all_texts.append(" <|endoftext|> ".join(convo_text))

    except Exception as e:
        print(f"An error occurred parsing the XML file: {e}")
        return False

    if not all_texts:
        print("No text data was extracted. Cannot proceed.")
        return False

    train_texts, val_texts = train_test_split(all_texts, test_size=0.1, random_state=42)

    train_file = os.path.join(output_dir, "train.txt")
    val_file = os.path.join(output_dir, "val.txt")

    with open(train_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(train_texts))

    with open(val_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(val_texts))

    print(f"Successfully created {train_file} and {val_file}.")
    return True

def main():
    CORPUS_XML = 'pan12-sexual-predator-identification-training-corpus-2012-05-01.xml'
    output_dir = "./fine_tuned_gpt2"
    model_name = "gpt2"

    corpus_path = get_file(CORPUS_XML)

    if not prepare_gpt2_data(corpus_path, output_dir):
        sys.exit()

    print("\n--- Starting GPT-2 Fine-Tuning ---")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.eos_token_id

    raw_datasets = load_dataset(
        "text",
        data_files={
            "train": os.path.join(output_dir, "train.txt"),
            "validation": os.path.join(output_dir, "val.txt")
        }
    )

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        max_steps=100,                 
        per_device_train_batch_size=8,
        save_steps=50,
        save_total_limit=1,
        logging_dir="./logs",
        logging_steps=20,              
        eval_strategy="no",      
        report_to="none",              
        fp16=True,                     
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"], 
        data_collator=data_collator,
    )

    print("Training GPT-2 model...")
    trainer.train()

    print(f"Saving fine-tuned GPT-2 model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("--- GPT-2 Training Complete ---")

if __name__ == "__main__":
    main()
