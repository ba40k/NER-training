# Product Name Extraction from Furniture Stores

## Overview
This project provides a solution for extracting product names from furniture store websites using Named Entity Recognition (NER). The system combines web scraping techniques with a fine-tuned BERT model to identify and extract product names from webpage content.

## Solution Architecture

### Key Components
1. **Web Scraper**: Extracts relevant text from furniture store product pages
2. **Custom NER Model**: Fine-tuned BERT model specialized in identifying furniture products
3. **API Layer**: FastAPI-based interface


## Implementation Details

### Model Training
- **Base Model**: `bert-base-uncased`
- **Training Data**: Custom BIO-tagged dataset of furniture product names
- **Labels**: `["O", "B-PRODUCT", "I-PRODUCT"]`
- **Tokens**: 223655 tokens for training and 44430 tokens for validation

### Key Features
- **Intelligent Text Extraction**: Focuses on product-relevant page sections
- **Subword Handling**: Special processing for BERT tokenization artifacts
- **Duplicate Removal**: Ensures clean output

## Metrics
Results: {'eval_loss': 0.01357267890125513,
 'eval_f1': 0.996487180831542,
  'eval_report': {'PRODUCT': {'precision': 0.9960171889739021, 'recall': 0.9969576164498531, 'f1-score': 0.996487180831542, 'support': 9532}, 
  'micro avg': {'precision': 0.9960171889739021, 'recall': 0.9969576164498531, 'f1-score': 0.996487180831542, 'support': 9532}, 
  'macro avg': {'precision': 0.9960171889739021, 'recall': 0.9969576164498531, 'f1-score': 0.996487180831542, 'support': 9532}, 
  'weighted avg': {'precision': 0.9960171889739021, 'recall': 0.9969576164498531, 'f1-score': 0.9964871808315421, 'support': 9532}}, 
  'eval_runtime': 73.2464, 'eval_samples_per_second': 104.674, 'eval_steps_per_second': 13.093, 'epoch': 9.0}


