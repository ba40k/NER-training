import requests
from bs4 import BeautifulSoup
from transformers import pipeline
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import cv2
import pytesseract
from PIL import Image
import io
import urllib.parse
import numpy as np
import os
from datetime import datetime

# 1. HuggingFace NER-pipeline 
hf_ner = pipeline(
    "ner",
    model="ner-product-model",
    tokenizer="ner-product-model",
    grouped_entities=True,
    aggregation_strategy="simple"
)

print("Модель NER инициализирована: dslim/bert-base-NER")

def extract_keywords_from_url(url: str) -> list:
    """Извлекает осмысленные части из URL"""
    parsed = urllib.parse.urlparse(url)
    path_parts = [p for p in parsed.path.split('/') if p and not p.isdigit()]
    query_parts = [p for p in parsed.query.split('&') if p] if parsed.query else []
    
    keywords = []
    for part in path_parts + query_parts:
        # Чистим и разбиваем слова
        clean = urllib.parse.unquote(part)
        clean = clean.replace('-', ' ').replace('_', ' ').strip()
        if len(clean) > 2:  # Игнорируем короткие фрагменты
            keywords.extend(clean.split())
    
    return list(set(keywords))  # Удаляем дубликаты

def fetch_text_from_url(
    url: str,
    output_dir: str = "screenshots",
    lang: str = "rus+eng"
) -> str:
    """Основная функция парсинга"""
    # Извлекаем ключевые слова из URL
    url_keywords = extract_keywords_from_url(url)
    
    # Настройка Selenium
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--window-size=1920,1080")
    
    driver = webdriver.Chrome(options=chrome_options)
    driver.get(url)
    time.sleep(3)

    # Делаем скриншот всей страницы
    total_height = driver.execute_script("return document.body.scrollHeight")
    driver.set_window_size(1920, total_height)
    
    os.makedirs(output_dir, exist_ok=True)
    screenshot_path = os.path.join(output_dir, f"screenshots/screenshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    driver.save_screenshot(screenshot_path)
    screenshot = driver.get_screenshot_as_png()
    driver.save_screenshot(screenshot_path)
    driver.quit()

    # OCR обработка
    img = cv2.imdecode(np.frombuffer(screenshot, np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray, lang=lang, config='--oem 3 --psm 6')

    # Очистка текста
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    # Объединяем URL-ключи и основной текст
    result = []
    if url_keywords:
        result.append("[Keywords from URL]")
        result.extend(url_keywords)
        result.append("\n[Page Content]")
    result.extend(lines)
    print(result)
    return '\n'.join(result)

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
    #print(text)

    entities = hf_ner(text)
    #print(entities)
    products = []

    for ent in entities:
        if (
            ent.get("entity_group", "").upper() == "PRODUCT"
            and ent.get("score", 0) >= 0.7
        ):
            products.append(ent["word"])
    return list(set(products))
