import os
import xml.etree.ElementTree as ET
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
import re
import sys
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import numpy as np



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
                print(f"Successfully received '{actual_filename}'.")
                return actual_filename
            except ImportError:
                pass
        else:
            print("="*80)
            print(f"ERROR: Dataset file '{filename_prompt}' not found in the local directory.")
            print("Please place the required file in the same directory as this script and run again.")
            print("="*80)
            sys.exit()
    print(f"Successfully found '{filename_prompt}'.")
    return filename_prompt



def get_predator_ids_from_txt(txt_file_path):
    predators = set()
    try:
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    predators.add(parts[0])
    except FileNotFoundError:
        print(f"Error: Predator truth file not found at {txt_file_path}")
    except Exception as e:
        print(f"An error occurred reading {txt_file_path}: {e}")
    return predators

def parse_conversations(corpus_xml_file, predator_ids=None):
    data = []
    try:
        tree = ET.parse(corpus_xml_file)
        root = tree.getroot()
        for conversation in root.findall('conversation'):
            for message in conversation.findall('message'):
                author_id_element = message.find('author')
                text_content_element = message.find('text')

                if author_id_element is not None and text_content_element is not None and author_id_element.text is not None:
                    author_id = author_id_element.text.strip()
                    text_content = text_content_element.text

                    if text_content:
                        clean_text = re.sub(r'<.*?>', '', text_content).strip()
                        row = {'text': clean_text}
                        if predator_ids is not None:
                            label = 1 if author_id in predator_ids else 0
                            row['label'] = label
                        data.append(row)
    except ET.ParseError as e:
        print(f"Warning: Could not parse {corpus_xml_file}: {e}")
    except Exception as e:
        print(f"An unexpected error occurred with file {corpus_xml_file}: {e}")

    return pd.DataFrame(data)

def main():
    TRAINING_CORPUS_XML = 'pan12-sexual-predator-identification-training-corpus-2012-05-01.xml'
    TRAINING_TRUTH_TXT = 'pan12-sexual-predator-identification-groundtruth-problem1.txt'

    training_corpus_path = get_file(TRAINING_CORPUS_XML)
    training_truth_path = get_file(TRAINING_TRUTH_TXT)

    print(f"Loading predator IDs from: {training_truth_path}")
    predator_author_ids = get_predator_ids_from_txt(training_truth_path)
    print(f"Found {len(predator_author_ids)} predator IDs.")

    print(f"Parsing conversations from: {training_corpus_path}")
    df = parse_conversations(training_corpus_path, predator_author_ids)

    if df.empty:
        print("\nTraining DataFrame is empty. Please check your file paths.")
        return

    print(f"Successfully loaded {len(df)} training messages.")
    print("\nOriginal Label distribution:\n", df['label'].value_counts())

    if df['label'].nunique() < 2:
        print("\nERROR: The training data contains only one class.")
        return

    predator_df = df[df['label'] == 1]
    non_predator_df = df[df['label'] == 0].sample(n=len(predator_df), random_state=42)

    df_balanced = pd.concat([predator_df, non_predator_df])
    print("Dataset has been balanced by undersampling.")
    print("\nBalanced Label distribution:\n", df_balanced['label'].value_counts())


    X = df_balanced['text'].values
    y = df_balanced['label'].values
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, ngram_range=(1, 2), stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    baseline_model = LinearSVC(class_weight='balanced', random_state=42, max_iter=2000)
    baseline_model.fit(X_train_tfidf, y_train)
    print("Baseline model training complete.")
    y_pred_baseline = baseline_model.predict(X_val_tfidf)
    print("\nBaseline Model Classification Report (on validation set):\n")
    print(classification_report(y_val, y_pred_baseline, target_names=['Non-Predator', 'Predator']))
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

    def encode_data(tokenizer, texts, max_length=128):
        input_ids = []
        attention_masks = []
        for text in texts:
            encoded = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            input_ids.append(encoded['input_ids'])
            attention_masks.append(encoded['attention_mask'])
        return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)

    train_inputs, train_masks = encode_data(tokenizer, X_train)
    val_inputs, val_masks = encode_data(tokenizer, X_val)

    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_val)

    batch_size = 16
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 1
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"Starting DistilBERT training for {epochs} epoch(s)...")
    for epoch_i in range(0, epochs):
        print(f'======== Epoch {epoch_i + 1} / {epochs} ========')
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            if step % 50 == 0 and not step == 0:
                print(f'  Batch {step} of {len(train_dataloader)}.')
            b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]
            model.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_train_loss = total_loss / len(train_dataloader)
        print(f"  Average training loss: {avg_train_loss}")
    model.eval()
    predictions, true_labels = [], []
    for batch in val_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_input_mask)
        logits = outputs[0]
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend(np.argmax(logits, axis=1).flatten())
        true_labels.extend(label_ids.flatten())

    print("\nDistilBERT Model Classification Report:\n")
    print(classification_report(true_labels, predictions, target_names=['Non-Predator', 'Predator']))

    
    output_dir = './model_save/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\nTraining complete and model saved.")


if __name__ == "__main__":
    main()
