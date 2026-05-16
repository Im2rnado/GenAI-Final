# Hybrid RAG Q&A Assistant

A robust, technical question-answering assistant that retrieves and generates answers to data science and machine learning questions using a hybrid search pipeline. This project maps to the milestones of the Advanced Machine Learning final project.

## Key Features
- **Lexical Search (BM25):** Fast exact-match keyword indexing powered by `rank_bm25`.
- **Semantic Search (MiniLM):** Deep vector embeddings generated via `sentence-transformers/all-MiniLM-L6-v2` and searched instantly using a CPU-based **FAISS** index.
- **Hybrid Fusion (RRF):** Results from lexical and semantic searches are combined using **Reciprocal Rank Fusion (RRF)** to optimize relevance.
- **RAG LLM Generation:** Augmented prompts are sent to highly capable models on OpenRouter (e.g., `openai/gpt-oss-120b:free`) to generate structured answers with inline citations.
- **Gradio Interactive UI:** An easy-to-use playground allowing live Q&A, source citation cards, and real-time visualization of evaluation metrics.

---

## Project Structure
```text
├── DataSet/
│   ├── top_ai_questions.json          # AI Stack Overflow questions corpus
│   └── top_datascience_questions.json  # Data Science questions corpus
├── Hybrid_RAG_QA.ipynb                # Main Demo Notebook with full milestones
├── Final Report.pdf               # Technical report covering methodology & metrics
└── README.md                          # Project log and quickstart instructions
```

---

## Quickstart & Setup

### 1. Prerequisites
Ensure you have Python 3.8+ installed. You can install all project requirements with pip:

```bash
pip install rank_bm25 sentence-transformers faiss-cpu datasets gradio plotly openai nltk pandas numpy
```

### 2. Running the System
Open the Jupyter Notebook and execute all cells sequentially:

```bash
jupyter notebook Hybrid_RAG_QA.ipynb
```

Once the final cell runs, it will launch a **Gradio** web interface. Click the local or public URL generated in the logs to open the interactive UI in your web browser.

---

## Evaluation Results
The system is evaluated against a technical ground-truth validation set at a cutoff of `k=10`:

| Method | MAP@10 | MRR@10 | nDCG@10 |
| :--- | :---: | :---: | :---: |
| **BM25** | 0.9292 | 1.00 | 0.9748 |
| **Semantic (MiniLM)** | 1.0000 | 1.00 | 1.0000 |
| **Hybrid-RRF** | 0.9775 | 1.00 | 0.9912 |