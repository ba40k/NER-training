# visualize_ner_results.py

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from seqeval.metrics import classification_report

# 1. Метки
labels = ["O", "B-PRODUCT", "I-PRODUCT"]
label2id = {lab: idx for idx, lab in enumerate(labels)}
id2label = {idx: lab for lab, idx in label2id.items()}

# 2. Чтение BIO-файла
def read_bio(filepath):
    sentences, tag_seqs = [], []
    with open(filepath, encoding="utf-8") as f:
        toks, tags = [], []
        for line in f:
            line = line.strip()
            if not line:
                if toks:
                    sentences.append(toks)
                    tag_seqs.append([label2id[t] for t in tags])
                    toks, tags = [], []
            else:
                token, tag = line.split()
                toks.append(token)
                tags.append(tag)
        if toks:  # на случай, если нет пустой строки в конце
            sentences.append(toks)
            tag_seqs.append([label2id[t] for t in tags])
    return sentences, tag_seqs

# 3. Загружаем валидацию
valid_tokens, valid_tag_ids = read_bio("valid.txt")

# 4. Модель и токенизатор
model_name = "ner-product-model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name).to(device)
model.eval()

# 5. Инференс и выравнивание
all_pred_ids = []

for tokens, true_ids in zip(valid_tokens, valid_tag_ids):
    # токенизируем без паддинга/транкейшна
    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        padding=False,
        truncation=False
    ).to(device)

    with torch.no_grad():
        logits = model(**enc).logits[0]        # [seq_len, num_labels]
    preds = torch.argmax(logits, dim=-1).cpu().tolist()  # [seq_len]

    word_ids = enc.word_ids(batch_index=0)  # длина == seq_len
    aligned = []
    prev_word_idx = None

    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        # берём только первый сабтокен каждого слова
        if word_idx != prev_word_idx:
            aligned.append(preds[idx])
        prev_word_idx = word_idx

    # отладочный ассерт: aligned должен совпадать с числом токенов
    if len(aligned) != len(tokens):
        raise ValueError(
            f"Mismatch: got {len(aligned)} predictions for {len(tokens)} tokens."
        )

    all_pred_ids.append(aligned)

# 6. Преобразуем в строковые метки
true_tag_seqs = [[id2label[i] for i in seq] for seq in valid_tag_ids]
pred_tag_seqs = [[id2label[i] for i in seq] for seq in all_pred_ids]

# 7. Печать детально
print("\n🔎 Результаты NER на validation:\n")
for toks, tr_seq, pr_seq in zip(valid_tokens, true_tag_seqs, pred_tag_seqs):
    for tok, true_lab, pred_lab in zip(toks, tr_seq, pr_seq):
        mark = "✔" if true_lab == pred_lab else "❌"
        print(f"{tok:20} | {true_lab:10} | {pred_lab:10} | {mark}")
    print("-" * 60)

# 8. Classification report
print("\n📊 Classification Report:\n")
print(classification_report(true_tag_seqs, pred_tag_seqs))
