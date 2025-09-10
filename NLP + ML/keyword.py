import os
import sys
import re
import xml.etree.ElementTree as ET
from typing import Set, Tuple

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

import torch
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_linear_schedule_with_warmup

def get_file(filename_prompt: str) -> str:
    is_in_colab = 'google.colab' in sys.modules
    if not os.path.exists(filename_prompt):
        if is_in_colab:
            try:
                from google.colab import files
                print(f"\nPlease upload the required dataset file: '{filename_prompt}'")
                uploaded = files.upload()
                if not uploaded:
                    print(f"\nERROR: File upload for '{filename_prompt}' failed.")
                    sys.exit(1)
                return list(uploaded.keys())[0]
            except ImportError:
                pass
        print("=" * 80)
        print(f"ERROR: Required file '{filename_prompt}' not found.")
        print("=" * 80)
        sys.exit(1)
    return filename_prompt


def get_predator_ids_from_txt(txt_file_path: str) -> Set[str]:
    predators = set()
    with open(txt_file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                predators.add(parts[0])
    return predators

def parse_conversations(corpus_xml_file: str, predator_ids: Set[str]) -> pd.DataFrame:
    data = []
    tree = ET.parse(corpus_xml_file)
    root = tree.getroot()
    for conversation in root.findall("conversation"):
        for message in conversation.findall("message"):
            author_id_elem = message.find("author")
            text_elem = message.find("text")
            if author_id_elem is None or text_elem is None:
                continue
            author_id = author_id_elem.text.strip()
            text = text_elem.text or ""
            text = re.sub(r"<.*?>", "", text).strip()
            label = 1 if author_id in predator_ids else 0
            data.append({"author": author_id, "text": text, "label": label})
    return pd.DataFrame(data)


def clean_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"http\S+", "", s)       
    s = re.sub(r"&\w+;", " ", s)        
    s = re.sub(r"[^a-z\s]", " ", s)      
    return re.sub(r"\s+", " ", s).strip()


def train_tfidf_svm(df: pd.DataFrame) -> Tuple[LinearSVC, TfidfVectorizer]:
    df = df.copy()
    df["clean_text"] = df["text"].fillna("").map(clean_text)
    X_train, X_val, y_train, y_val = train_test_split(
        df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=5, stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)

    model = LinearSVC(class_weight="balanced", max_iter=2000)
    model.fit(X_train_tfidf, y_train)

    y_pred = model.predict(X_val_tfidf)
    print("\n=== TF-IDF + SVM Report ===")
    print(classification_report(y_val, y_pred, target_names=["Non-Predator","Predator"]))

    return model, vectorizer


def encode_texts(tokenizer, texts, max_len=128):
    input_ids, masks = [], []
    for text in texts:
        enc = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        input_ids.append(enc["input_ids"])
        masks.append(enc["attention_mask"])
    return torch.cat(input_ids, dim=0), torch.cat(masks, dim=0)

def stratified_sample(df: pd.DataFrame, n_sample: int = 5000, random_state: int = 42) -> pd.DataFrame:
    if n_sample >= len(df):
        return df.copy()
    frac = n_sample / len(df)
    sample, _ = train_test_split(df, train_size=frac, stratify=df["label"], random_state=random_state)
    return sample.reset_index(drop=True)

def oversample_minority(df: pd.DataFrame, target_minority_ratio: float = 0.25, random_state: int = 42) -> pd.DataFrame:
    df = df.copy()
    counts = df["label"].value_counts()
    if len(counts) < 2:
        return df
    n_total = len(df)
    n_target_min = int(target_minority_ratio * n_total)
    n_min = counts.get(1, 0)
    if n_min >= n_target_min:
        return df  
    
    df_min = df[df["label"] == 1]
    df_maj = df[df["label"] == 0]
    n_needed = n_target_min - n_min
    if n_needed <= 0:
        return df
    min_upsampled = df_min.sample(n=n_needed, replace=True, random_state=random_state)
    df_new = pd.concat([df_maj, df_min, min_upsampled]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df_new


def train_distilbert(df: pd.DataFrame,
                     n_sample: int = 5000,
                     target_minority_ratio: float = 0.25,
                     epochs: int = 3,
                     batch_size: int = 16,
                     max_length: int = 128,
                     lr: float = 2e-5):
   
    df_small = stratified_sample(df, n_sample=n_sample)
    print(f"Initial sampled size: {len(df_small)} (minority count = {df_small['label'].sum()})")

    
    df_balanced = oversample_minority(df_small, target_minority_ratio=target_minority_ratio)
    print(f"After oversampling minority: total={len(df_balanced)}, minority={df_balanced['label'].sum()}")

    
    df_balanced["clean_text"] = df_balanced["text"].fillna("").map(clean_text)
    X_train, X_val, y_train, y_val = train_test_split(
        df_balanced["clean_text"], df_balanced["label"], test_size=0.2, random_state=42, stratify=df_balanced["label"]
    )

    
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    
    train_inputs, train_masks = encode_texts(tokenizer, X_train, max_len=max_length)
    val_inputs, val_masks = encode_texts(tokenizer, X_val, max_len=max_length)

    train_labels = torch.tensor(y_train.values, dtype=torch.long)
    val_labels = torch.tensor(y_val.values, dtype=torch.long)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    val_data = TensorDataset(val_inputs, val_masks, val_labels)

    train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=batch_size)
    val_dataloader = DataLoader(val_data, sampler=SequentialSampler(val_data), batch_size=batch_size)

    
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    
    counts = np.bincount(y_train.astype(int))
   
    if len(counts) < 2:
        class_weights = torch.tensor([1.0, 1.0], dtype=torch.float)
    else:
        
        inv_freq = np.array([ (1.0 / max(1, counts[0])), (1.0 / max(1, counts[1])) ], dtype=np.float32)
        
        inv_freq = inv_freq / inv_freq.sum() * 2.0
        class_weights = torch.tensor(inv_freq, dtype=torch.float)
    class_weights = class_weights.to(device)
    loss_fn = CrossEntropyLoss(weight=class_weights)

    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]
            model.zero_grad()
            outputs = model(b_input_ids, attention_mask=b_input_mask, return_dict=True)
            logits = outputs.logits  
            loss = loss_fn(logits, b_labels)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        avg_loss = total_loss / max(1, len(train_dataloader))
        print(f"Epoch {epoch+1}/{epochs} - avg_train_loss: {avg_loss:.4f}")

        
        model.eval()
        preds, truths = [], []
        with torch.no_grad():
            for batch in val_dataloader:
                b_input_ids, b_input_mask, b_labels = [t.to(device) for t in batch]
                outputs = model(b_input_ids, attention_mask=b_input_mask, return_dict=True)
                logits = outputs.logits.detach().cpu().numpy()
                labels = b_labels.to("cpu").numpy()
                preds.extend(np.argmax(logits, axis=1).tolist())
                truths.extend(labels.tolist())
        print(f"\n=== Validation after epoch {epoch+1} ===")
        print(classification_report(truths, preds, target_names=["Non-Predator", "Predator"]))

    return model, tokenizer


def main():
    TRAINING_XML = "pan12-sexual-predator-identification-training-corpus-2012-05-01.xml"
    TRAINING_TRUTH = "pan12-sexual-predator-identification-groundtruth-problem1.txt"

    
    DISTILBERT_SAMPLE_SIZE = 5000          
    DISTILBERT_EPOCHS = 3
    DISTILBERT_BATCH = 16
    TARGET_MINORITY_RATIO = 0.25          
    MAX_LEN = 128
    LR = 2e-5

    xml_path = get_file(TRAINING_XML)
    truth_path = get_file(TRAINING_TRUTH)

    predator_ids = get_predator_ids_from_txt(truth_path)
    df = parse_conversations(xml_path, predator_ids)
    print(f"Loaded {len(df)} messages. Predator messages: {df['label'].sum()}")

    
    tfidf_model, vectorizer = train_tfidf_svm(df)

    
    distilbert_model, tokenizer = train_distilbert(
        df,
        n_sample=DISTILBERT_SAMPLE_SIZE,
        target_minority_ratio=TARGET_MINORITY_RATIO,
        epochs=DISTILBERT_EPOCHS,
        batch_size=DISTILBERT_BATCH,
        max_length=MAX_LEN,
        lr=LR
    )

    
    os.makedirs("saved_models", exist_ok=True)
    import joblib
    joblib.dump((tfidf_model, vectorizer), "saved_models/tfidf_svm.pkl")
    distilbert_model.save_pretrained("saved_models/distilbert_model")
    tokenizer.save_pretrained("saved_models/distilbert_model")
    print("Models saved to 'saved_models/'")

if __name__ == "__main__":
    main()
