from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
from seqeval.metrics import f1_score, classification_report
import numpy as np
import torch

# Метки
labels = ["O", "B-PRODUCT", "I-PRODUCT"]
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

# Чтение BIO-файла с улучшенной обработкой
def read_bio_file(filepath):
    tokens_list, tags_list = [], []
    with open(filepath, "r", encoding="utf-8") as f:
        tokens, tags = [], []
        for line in f:
            line = line.strip()
            if not line:
                if tokens:
                    tokens_list.append(tokens)
                    tags_list.append(tags)
                    tokens, tags = [], []
            else:
                # Обработка случаев с дополнительными пробелами
                parts = line.split()
                token = parts[0]
                tag = parts[-1] if len(parts) > 1 else "O"
                tokens.append(token)
                tags.append(label2id.get(tag, 0))
        if tokens:
            tokens_list.append(tokens)
            tags_list.append(tags)
    return tokens_list, tags_list

# Загрузка данных
train_tokens, train_labels = read_bio_file("train.txt")
valid_tokens, valid_labels = read_bio_file("valid.txt")

dataset = DatasetDict({
    "train": Dataset.from_dict({"tokens": train_tokens, "ner_tags": train_labels}),
    "validation": Dataset.from_dict({"tokens": valid_tokens, "ner_tags": valid_labels}),
})

# Токенизация с исправленным выравниванием меток
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding=True,
        is_split_into_words=True,
        max_length=128,
        return_tensors="pt"
    )
    
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Специальные токены получают -100
            if word_idx is None:
                label_ids.append(-100)
            # Новое слово
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # Подтокены одного слова
            else:
                current_label = label[word_idx]
                # Преобразуем B -> I для продолжения слова
                if id2label[current_label] == "B-PRODUCT":
                    label_ids.append(label2id["I-PRODUCT"])
                else:
                    label_ids.append(current_label)
            previous_word_idx = word_idx
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

# Применяем с батчингом для ускорения
tokenized = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    batch_size=32
)

# Модель с улучшенной инициализацией
model = AutoModelForTokenClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

#  Расширенные метрики
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []
    
    for i in range(len(predictions)):
        preds = [id2label[p] for p, l in zip(predictions[i], labels[i]) if l != -100]
        labs = [id2label[l] for l in labels[i] if l != -100]
        
        if len(preds) > 0:
            true_predictions.append(preds)
            true_labels.append(labs)
    
    return {
        "f1": f1_score(true_labels, true_predictions),
        "report": classification_report(true_labels, true_predictions, output_dict=True)
    }

# Улучшенные параметры обучения
training_args = TrainingArguments(
    output_dir="ner-product-model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=8,  # Уменьшил для стабильности
    per_device_eval_batch_size=8,
    num_train_epochs=15,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer),
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Запуск с проверкой данных
if __name__ == "__main__":
    # Проверка выравнивания меток
    sample = tokenized["train"][0]
    print("\nПроверка выравнивания:")
    print("Токены:", tokenizer.convert_ids_to_tokens(sample["input_ids"]))
    print("Метки:", [id2label.get(l, "UNK") for l in sample["labels"]])
    
    # Обучение
    trainer.train()
    
    # Сохранение лучшей модели
    trainer.save_model("ner-product-model")
    print(" Обучение завершено. Модель сохранена")
    print("Итоговые метрики:", trainer.evaluate())