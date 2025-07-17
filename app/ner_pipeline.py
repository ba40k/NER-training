# app/ner_pipeline.py

import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# 1. HuggingFace NER-pipeline (единожды инициализируется при импорте модуля)
hf_ner = pipeline(
    "ner",
    model="/home/radamir/EvidantixTask/EvidantixInternshipTask/ner-product-model",
    tokenizer="/home/radamir/EvidantixTask/EvidantixInternshipTask/ner-product-model",
    grouped_entities=True,
    aggregation_strategy="simple"
)

print("Модель NER инициализирована: dslim/bert-base-NER")


def fetch_text_from_url(url: str) -> str:
    import requests
    from bs4 import BeautifulSoup

    resp = requests.get(url, timeout=10)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.content, "html.parser")
    texts = []

    # Удалим лишние теги: скрипты, стили, скрытые элементы
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    for hidden in soup.select("[aria-hidden='true'], [style*='display:none'], [hidden]"):
        hidden.decompose()

    # Название товара
    title = soup.select_one("h1.product-single__title")
    if title:
        texts.append(title.get_text(strip=True))

    # Описание товара
    desc = soup.select_one("div.product-single__description")
    if desc:
        texts.append(desc.get_text(separator=" ", strip=True))

    # Поиск внутри `<main>` или `.product-detail` — контекстно ближе к товару
    container = soup.select_one("main, .product-detail, .product-page")
    if container:
        for tag in container.find_all(["h1", "h2", "p", "span", "li"]):
            txt = tag.get_text(separator=" ", strip=True)
            if txt:
                texts.append(txt)

    # Финальная очистка: удалим повторяющийся текст
    clean_texts = list(dict.fromkeys(texts))  # сохраняем порядок, удаляем дубликаты

    print("Извлечённые фрагменты:\n", "\n".join(clean_texts[:5]))

    return " ".join(clean_texts)


def detokenize(tokens: list[str]) -> list[str]:
    """
    Склеивает BERT-подтокены в цельные слова.
    Пример:
    ["Re", "##cliner", "Table"] → ["Recliner", "Table"]
    """
    words = []
    current = ""

    for token in tokens:
        if token.startswith("##"):
            current += token[2:]
        else:
            if current:
                words.append(current)
            current = token
    if current:
        words.append(current)

    return words

def extract_products_hf(text: str) -> list[str]:
    """
    Прогоняет текст через HF-NER и возвращает список
    уникальных сущностей с группой “PRODUCT”.
    """
    entities = hf_ner(text)
    products = []

    for ent in entities:
        # HF возвращает ключи “entity_group” и “word”
        if ent.get("entity_group", "").upper() == "PRODUCT":
            products.append(ent["word"])

    return list(set(detokenize(products)))
