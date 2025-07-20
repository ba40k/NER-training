import random
import os
from tqdm import tqdm

# 📁 Пути
DATA_DIR = "token_generation"
TRAIN_PATH = os.path.join(DATA_DIR, "tokens.txt")
VALID_PATH = os.path.join(DATA_DIR, "valid.txt")

def read_lines(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def insert_noise(tokens, noises, noise_ratio=0.6, max_noise_tokens=4):
    clean_tokens = [("P", token) for token in tokens]
    if not noises or random.random() > noise_ratio:
        return clean_tokens

    num_noises = random.randint(2, max_noise_tokens)
    noise_tokens = random.choices(noises, k=num_noises)

    positions = random.choices(["start", "middle", "end"], k=len(noise_tokens))

    noisy_phrase = clean_tokens.copy()
    for noise, pos in zip(noise_tokens, positions):
        if pos == "start":
            noisy_phrase = [("O", noise)] + noisy_phrase
        elif pos == "middle":
            idx = random.randint(1, len(noisy_phrase)-1) if len(noisy_phrase) > 1 else 0
            noisy_phrase = noisy_phrase[:idx] + [("O", noise)] + noisy_phrase[idx:]
        else:  # end
            noisy_phrase += [("O", noise)]

    return noisy_phrase


def generate_phrase(articles, adjectives, products, conjunctions, sub_products):
    parts = [
        random.choice(articles),
        random.choice(adjectives),
        random.choice(products)
    ]

    if random.random() < 0.6 and conjunctions and sub_products:
        parts += [random.choice(conjunctions), random.choice(sub_products)]

    return parts

def write_bio(file, phrase_tokens):
    previous_was_noise = True  # начнем с B-PRODUCT

    for tag, token in phrase_tokens:
        if tag == "O":
            file.write(f"{token}\tO\n")
            previous_was_noise = True
        elif tag == "P":
            bio_tag = "B-PRODUCT" if previous_was_noise else "I-PRODUCT"
            file.write(f"{token}\t{bio_tag}\n")
            previous_was_noise = False
    file.write("\n")


def main(sample_size=10000, train_ratio=0.8, noise_ratio=0.6):
    adjectives = read_lines("adjectives.txt")
    articles = read_lines("articles.txt")
    conjunctions = read_lines("conjunctions.txt")
    products = read_lines("products.txt")
    sub_products = read_lines("sub_products.txt")
    noises = read_lines("noises.txt")

    phrases = []
    for _ in range(sample_size):
        phrase = generate_phrase(articles, adjectives, products, conjunctions, sub_products)
        phrase_with_noise = insert_noise(phrase, noises, noise_ratio)
        phrases.append(phrase_with_noise)

    # 🎲 Разделение
    random.shuffle(phrases)
    split = int(len(phrases) * train_ratio)
    train_phrases = phrases[:split]
    valid_phrases = phrases[split:]

    os.makedirs(DATA_DIR, exist_ok=True)

    with open(TRAIN_PATH, "w", encoding="utf-8") as train_f, \
         open(VALID_PATH, "w", encoding="utf-8") as valid_f:

        for phrase_tokens in tqdm(train_phrases, desc="Writing tokens.txt"):
            write_bio(train_f, phrase_tokens)

        for phrase_tokens in tqdm(valid_phrases, desc="Writing valid.txt"):
            write_bio(valid_f, phrase_tokens)

    print(f"\n✅ TRAIN: {len(train_phrases)} phrases → {TRAIN_PATH}")
    print(f"✅ VALID: {len(valid_phrases)} phrases → {VALID_PATH}")

if __name__ == "__main__":
    main()
