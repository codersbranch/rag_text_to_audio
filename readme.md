# AI Voice PDF Chatbot using Ollama, Whisper, Kokoro & Streamlit

An end-to-end offline AI PDF chatbot built with Streamlit, Ollama,
FAISS, Whisper and Kokoro.

## Features

-   Upload PDFs
-   Automatic FAISS vector indexing
-   Tag-based document management
-   Text & Voice chat
-   Whisper Speech-to-Text
-   Ollama LLM
-   Kokoro Text-to-Speech
-   Offline RAG

## Installation

``` bash
git clone https://github.com/codersbranch/rag_text_to_audio
cd rag_text_to_audio
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
```


## Install Ollama

Download and install Ollama from:

https://ollama.com/download

Verify installation:
``` bash
ollama --version
```
Download the DeepSeek model

Start the Ollama service, then pull the model:
``` bash
ollama pull deepseek-r1:1.5b
```
Verify installed models:
``` bash
ollama list
```
(Optional) Test the model:
``` bash
ollama run deepseek-r1:1.5b
```




## Run

``` bash
streamlit run app.py
```

## Workflow

1.  Upload PDFs
2.  Create embeddings
3.  Select tag
4.  Ask using text or voice
5.  Whisper transcribes
6.  FAISS retrieves context
7.  Ollama generates answer
8.  Kokoro speaks the answer

## Models

-   all-MiniLM-L6-v2
-   Whisper Medium
-   deepseek-r1:1.5b
-   Kokoro

## Future Improvements

-   Chat history
-   Source citations
-   Docker
-   GPU support
