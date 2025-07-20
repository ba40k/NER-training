import requests
from bs4 import BeautifulSoup
from transformers import pipeline

# 1. HuggingFace NER-pipeline 
hf_ner = pipeline(
    "ner",
    model="/home/radamir/Загрузки/EvidantixInternshipTask-main/ner-product-model",
    tokenizer="/home/radamir/Загрузки/EvidantixInternshipTask-main/ner-product-model",
    grouped_entities=True,
    aggregation_strategy="simple"
)

print("Модель NER инициализирована: dslim/bert-base-NER")


def fetch_text_from_url(url: str) -> str:
    import requests
    from bs4 import BeautifulSoup
    from bs4.element import Comment, Declaration
    from urllib.parse import urlparse, unquote

    # Загружаем страницу
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")

    # Удаляем неинформативные теги
    for tag in soup(["script", "style", "noscript", "meta", "link", "svg", "img"]):
        tag.decompose()

    # Удаляем скрытые элементы
    for hidden in soup.select("""
        [style*='display:none'], 
        [style*='visibility:hidden'], 
        [aria-hidden='true'], 
        [hidden], 
        [style*='opacity:0'], 
        [style*='width:0'], 
        [style*='height:0'],
        [style*='position:absolute'][style*='left:-9999px'],
        .hidden,
        .invisible,
        .visually-hidden
    """):
        hidden.decompose()

    # Функция для проверки видимости элемента
    def is_visible(element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, (Comment, Declaration)):
            return False
        return True

    # Извлекаем видимый текст
    page_texts = []
    for tag in soup.find_all(string=True):
        if not is_visible(tag):
            continue
            
        txt = tag.strip()
        if txt and len(txt) > 2:  # избегаем обрывков вроде ",", "OK"
            # Проверяем, что родительский элемент не скрыт
            parent = tag.parent
            parent_hidden = any(
                parent.has_attr(attr) and parent[attr] == 'true' 
                for attr in ['aria-hidden', 'hidden']
            ) or any(
                style in parent.get('style', '').lower()
                for style in ['display:none', 'visibility:hidden', 'opacity:0']
            )
            
            if not parent_hidden:
                page_texts.append(txt)

    # Извлекаем осмысленные части из URL и помещаем их в начало
    url_parts = urlparse(url).path.split("/")
    url_words = []
    for part in url_parts:
        clean = unquote(part.strip().replace("-", " ").replace("_", " "))
        if clean and not clean.isdigit() and len(clean) > 2:
            url_words.append(clean)

    # Объединяем: сначала слова из URL, затем основной текст
    combined_texts = url_words + page_texts

    # Удаляем дубликаты, сохраняя порядок (первое вхождение остается)
    clean_texts = []
    seen = set()
    for text in combined_texts:
        if text not in seen:
            seen.add(text)
            clean_texts.append(text)

    return "\n".join(clean_texts)

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
    print(text)

    entities = hf_ner(text)
    print(entities)
    products = []

    for ent in entities:
        if (
            ent.get("entity_group", "").upper() == "PRODUCT"
            and ent.get("score", 0) >= 0.7
        ):
            products.append(ent["word"])
    return list(set(products))
