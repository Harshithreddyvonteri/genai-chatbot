# Design Decisions & Analysis

This document summarizes the **key design decisions** and **analysis** that shaped the GitLab Handbook Chatbot.

---

## 1. Dataset Versions

We created 3 versions of the dataset by scraping the GitLab Handbook:
| Version | Docs | Words    | Tokens   |
|---------|------|----------|----------|
| v1      | 20   | ~330k    | ~600k    |
| v2      | 100  | ~1.6M    | ~3M      |
| v3      | 300  | ~4.3M    | ~8M      |

**Decision:**  
- **v2** was selected as the base dataset for the chatbot:
  - **v1:** Too small and lacked adequate coverage.
  - **v3:** Too large, noisy, and introduced duplicates even after cleaning.
  - **v2:** Struck the right balance between coverage and retrieval speed.

---

## 2. Preprocessing

The raw data contained menus, duplicate content, and HTML artifacts.  
We built a **preprocessor pipeline** that:
- Removed duplicates and irrelevant sections.
- Stripped HTML tags and special characters.
- Split content into **~500 token chunks** (ideal for embeddings).

**Outcome:**  
Dataset size reduced by **~80%**, retaining clean, meaningful content.

---

## 3. Embeddings

We chose **`all-MiniLM-L6-v2`** from Sentence Transformers due to:
- **Speed:** Lightweight and fast inference on CPU.
- **Quality:** Performs well for semantic search tasks.
- **Open-source:** No API costs or restrictions.

---

## 4. Vector Database

We initially used **ChromaDB** during development.  
**Final Choice:** **FAISS**
- Stores all embeddings in a **single `.pkl` file** for easy deployment.
- Faster for in-memory search on mid-sized datasets.
- Eliminates need for persistent database hosting (critical for Streamlit Cloud).

---

## 5. LLM Choice & Fallback Logic

We selected **Google Gemini 1.5 Flash**:
- **Free tier:** 1500 API calls/day.
- **Low latency:** Faster responses than GPT-4 for this scale.
- **Reasoning fallback:**  
  - If retrieved handbook context is weak (e.g., <50 words), Gemini answers using its reasoning.

---

## 6. UI & UX Design

- **Streamlit** was chosen for rapid prototyping and hosting ease.
- Built a **ChatGPT-like interface**:
  - Dark mode.
  - Right-aligned bubbles for user questions.
  - Left-aligned bubbles for bot answers.
- **Fixed input clearing bug** by using `on_change` callbacks.

---

## 7. Future Enhancements
- Merge **best sections of v2 + v3** for better coverage.
- Add **streaming answers** for a real-time feel.
- Introduce **scrollable chat with auto-scroll** like ChatGPT.
- Add **metrics** (response time, number of calls).

---
