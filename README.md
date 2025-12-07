# 🔍 Natural Language Opinion Search Engine

A sophisticated opinion-based search system for product reviews using NLP techniques, from basic Boolean retrieval to state-of-the-art BERT semantic search.

## 📋 Project Overview

This project implements and compares three progressively advanced methods for searching product reviews based on aspect-opinion queries (e.g., "audio quality: poor", "wifi signal: strong"). Built as part of COSC 4397 Natural Language Processing coursework at the University of Houston.

### Dataset
- **Source**: Amazon product reviews (electronics & software)
- **Size**: 153,247 unique vocabulary terms across thousands of reviews
- **Metadata**: Star ratings, helpfulness votes, review titles, customer IDs

## 🎯 Query Format

All queries follow the structure: `[aspect: opinion]`

**Example queries:**
- `audio quality: poor`
- `wifi signal: strong`  
- `mouse button: click problem`
- `gps map: useful`
- `image quality: sharp`

## 🚀 Methods Implemented

### **Baseline: Boolean Search**
Traditional information retrieval using inverted index.

**Approach:**
- Preprocesses reviews (lemmatization, stopword removal)
- Builds inverted index for O(1) term lookup
- Supports AND/OR operators for aspect and opinion terms

**Strengths:** Fast, simple, interpretable  
**Limitations:** No semantic understanding, struggles with synonyms

---

### **Method 1: Rating-Filtered Boolean Search**
Enhances baseline with sentiment-aware filtering using the Hu & Liu opinion lexicon (KDD 2004).

**Approach:**
1. Performs Boolean retrieval
2. Classifies opinion polarity (positive/negative) using 6,789-word lexicon
3. Filters results by star rating consistency:
   - Positive opinions → ratings > 3 stars
   - Negative opinions → ratings ≤ 3 stars

**Novel contribution:** Cross-validates textual opinions with numerical ratings

**Strengths:** Higher precision, reduces false positives  
**Limitations:** Still keyword-dependent, ignores context

---

### **Method 2: BERT Semantic Search (Novel)**
State-of-the-art approach using transformer-based embeddings for semantic matching.

**Approach:**
1. Pre-computes BERT sentence embeddings (all-MiniLM-L6-v2) for entire corpus
2. Constructs natural language query: `"The {aspect} is {opinion}"`
3. Encodes query using same BERT model
4. Computes cosine similarity between query and all sentence embeddings
5. Retrieves semantically similar sentences (threshold: 0.70)
6. Validates with rating consistency

**Why this is powerful:**
- Finds semantically related reviews even without exact keyword matches
- Example: Query "audio quality: poor" matches "The sound was terrible and unclear" (no exact keywords!)
- Captures contextual meaning beyond bag-of-words

**Novel contributions:**
- Sentence-level semantic matching aggregated to document-level results
- Hybrid approach combining neural embeddings with sentiment validation
- Sub-2-second query time on full corpus using vectorized similarity computation

**Strengths:** High recall, semantic understanding, synonym-aware  
**Trade-offs:** Requires pre-computed embeddings (~500MB), GPU-accelerated preferred

---

## 📊 Results & Evaluation

Each method is evaluated on:
- **# Retrieved**: Total documents returned
- **# Relevant**: Manually validated relevant documents  
- **Precision**: Relevant / Retrieved

### Key Findings
- **Baseline**: High recall, lower precision (many false positives)
- **Method 1**: Improved precision through rating filtering (~15-30% reduction in results)
- **Method 2**: Best semantic relevance, finds reviews baseline methods miss

*Full results table and manual relevance judgments available in project report.*

## 🛠️ Technical Stack

- **Python 3.13**
- **NLP Libraries**: NLTK, sentence-transformers, scikit-learn
- **Deep Learning**: PyTorch, Hugging Face Transformers
- **Data Processing**: Pandas, NumPy

## 📁 Project Structure

```
nlp-engine/
├── Codes/
│   ├── NLProject.ipynb           # Main implementation notebook
│   ├── reviews_segment.pkl       # Amazon reviews dataset
│   ├── data.pkl                  # Pre-computed BERT embeddings (500MB)
│   ├── positive-words.txt        # Hu & Liu positive lexicon (2,006 words)
│   ├── negative-words.txt        # Hu & Liu negative lexicon (4,783 words)
├── Outputs/
│   ├── Baseline/                 # Boolean search results (txt files)
│   └── AdvancedModel/
│       ├── Method_1/             # Rating-filtered results
│       └── Method_2/             # BERT semantic search results
└── README.md                     # This file
```

## 🚦 Quick Start

### Prerequisites
```bash
pip install pandas numpy nltk sentence-transformers scikit-learn torch
```

### Generate BERT Embeddings (One-time, ~15 minutes)
```python
python create_embeddings.py
```

### Run Queries
```python
from NLProject import boolean_baseline, m1, m2

# Baseline
results = boolean_baseline("audio quality", "poor", "OR", "", "AND", "output.txt")

# Method 1: Rating Filter
results = m1("audio quality", "poor", "OR", "", "AND", "output.txt")

# Method 2: BERT Semantic Search
results = m2("audio quality", "poor", "output.txt")
```

## 🎓 Academic Context

**Course**: COSC 4397 - Natural Language Processing  
**Institution**: University of Houston  
**Instructor**: Dr. Arjun Mukherjee  
**Semester**: Fall 2025

### Key Concepts Demonstrated
- Information Retrieval (Boolean model, inverted index)
- Lexicon-based Sentiment Analysis
- Transfer Learning (pre-trained BERT)
- Semantic Similarity (cosine distance in embedding space)
- Evaluation Metrics (precision, recall)

## 📚 References

1. Hu, M., & Liu, B. (2004). Mining and summarizing customer reviews. *KDD '04*
2. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. *EMNLP 2019*
3. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*

## 🤝 Acknowledgments

- **Navid (TA)**: BERT embeddings infrastructure and starter code
- **Hu & Liu**: Opinion lexicon dataset
- **Sentence-Transformers team**: all-MiniLM-L6-v2 model

## 📄 License

Academic project - Educational use only

---

**Author**: Talha Mohammed  
**Contact**: [mohammedtalha290@gmail.com]  
**Date**: December 6th 2025
