# Chat With Your Documents — RAG Project

A local-first Retrieval-Augmented Generation (RAG) demo. Upload PDF, DOCX, TXT, or Markdown documents, then ask questions grounded in their content.

## Features

- Extracts text from PDF, Word, TXT, and Markdown files
- Splits text into overlapping chunks
- Generates semantic embeddings with `sentence-transformers`
- Stores and searches vectors in persistent local ChromaDB
- Sends retrieved context to an OpenAI chat model
- Displays retrieved chunks so answers can be checked against the source
- Instructs the model to say when the documents do not contain an answer

## Architecture

```text
Upload documents
      ↓
Text extraction
      ↓
Chunking (900 characters, 150 overlap)
      ↓
SentenceTransformer embeddings
      ↓
ChromaDB vector index
      ↓
Question embedding → semantic retrieval (top-k chunks)
      ↓
OpenAI chat model + retrieved context
      ↓
Answer with source filenames
```

## Requirements

- Python 3.10 or newer
- An OpenAI API key and API access for the selected model
- Internet access the first time the embedding model is downloaded

## Run locally

### 1. Create and activate a virtual environment

**Windows**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the app

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit. Paste your API key into the sidebar, upload documents, click **Process documents**, then ask a question.

Alternatively, set `OPENAI_API_KEY` in your environment before starting the app. Never commit real API keys to GitHub.

## Suggested test

1. Create a text file containing: `The Orion project deadline is 18 November. The project owner is Kavya.`
2. Upload and process the file.
3. Ask: `When is the Orion project deadline?`
4. Confirm that the answer says **18 November** and cites the uploaded filename.
5. Ask: `What is the budget for Orion?`
6. Confirm that the app says the documents do not contain enough information rather than inventing a budget.

## Important notes

- Scanned PDFs may not contain extractable text; OCR is not included in this version.
- ChromaDB persists the local index in `.chroma_db/`. Use **Clear indexed documents** to remove it.
- The embedding model runs locally, but retrieved document text is sent to the configured OpenAI API to generate answers. Do not upload confidential material unless permitted.
- This is a learning project, not a production-secure multi-user application.
- Retrieval quality depends on document quality, chunk size, embedding model, and the wording of the question.

## Improvements to try

- Add OCR for scanned PDFs
- Add reranking and metadata filters
- Display similarity scores and improve evaluation
- Add streaming responses and conversation history
- Add a citation-to-page mapping for PDFs
- Deploy with authentication and per-user storage
