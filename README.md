# 🧾 Labeling Requirements Extraction Pipeline

This repository contains a complete machine learning pipeline for extracting and structuring food labeling requirements from regulatory PDF documents. The pipeline uses both traditional ML (TF-IDF + Logistic Regression) and modern deep learning (BERT) approaches.

## 📋 Pipeline Overview

The pipeline consists of 11 sequential scripts that process PDF documents through text extraction, preprocessing, model training, prediction, and structured output generation. Each script depends on the outputs of the previous steps.

## 📁 Directory Structure

data/
├── 1_raw/                     # Input PDF files  
├── 2_extracted/              # Extracted text files  
├── 3_processed/              # Preprocessed and cleaned text  
├── 4_sections/               # Labeling-specific sections  
├── 5_dataset/                # Labeled training dataset  
├── 6_predictions/            # Traditional ML predictions  
├── 7_predictions_bert/       # BERT model predictions  
├── 8_pdf_model_extraction/   # Direct PDF processing with BERT  
└── 9_structured/             # Final structured requirements  

models/  
├── traditional/              # TF-IDF + Logistic Regression models  
└── bert_finetuned/           # Fine-tuned BERT model

## ⚙️ Execution Flow

### Phase 1: Data Preparation & Preprocessing

1. **Text Extraction**
```bash
python 01_extract_text.py
