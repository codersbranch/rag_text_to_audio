import faiss
import pickle
import streamlit as st
import ollama
import time
import tempfile
import io
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import re
import whisper
from kokoro import KPipeline
import soundfile as sf
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ----------------------
# Load all PDFs
# ----------------------
def load_all_pdfs(pdf_dir):
    all_docs = []

    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):

            pdf_path = os.path.join(pdf_dir, filename)

            loader = PyPDFLoader(pdf_path)
            docs = loader.load()

            all_docs.extend(docs)

    return all_docs


# ----------------------
# Split documents
# ----------------------
def split_documents(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(docs)


# ----------------------
# Create Vector Store
# ----------------------
def create_vector_index(split_docs):

    embedder = get_embedder()

    texts = [doc.page_content for doc in split_docs]

    embeddings = embedder.encode(
        texts,
        show_progress_bar=True
    )

    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)

    index.add(
        embeddings.astype(np.float32)
    )

    return index, texts


# ----------------------
# Save Vector Store
# ----------------------
def save_index(index, texts, output_folder):

    os.makedirs(output_folder, exist_ok=True)

    faiss.write_index(
        index,
        os.path.join(output_folder, "vectors.index")
    )

    with open(
        os.path.join(output_folder, "metadata.pkl"),
        "wb"
    ) as f:

        pickle.dump(texts, f)


# ----------------------
# Process Uploaded PDFs
# ----------------------
def process_uploaded_folder(tag):

    pdf_folder = os.path.join("uploaded_pdfs", tag)

    docs = load_all_pdfs(pdf_folder)

    if len(docs) == 0:
        return False

    split_docs = split_documents(docs)

    index, texts = create_vector_index(split_docs)

    output_folder = os.path.join(
        "vector_data",
        f"{tag}_index"
    )

    save_index(index, texts, output_folder)

    return True


def load_vector_store(index_path, docs_path):
    # Load FAISS index
    index = faiss.read_index(index_path)

    # Load documents
    with open(docs_path, "rb") as f:
        documents = pickle.load(f)

    # Reuse the cached embedder instead of constructing a new one every call
    embedder = get_embedder()
    return index, documents, embedder


# ----------------------
# Retrieval function
# ----------------------
def retrieve_context(query, embedder, index, documents, k=3):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding.astype(np.float32), k)
    return [documents[i] for i in indices[0]]


def remove_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)


# ----------------------
# Generate answer with Ollama
# ----------------------
def generate_answer_with_ollama(query, context):
    formatted_context = "\n".join(context)

    prompt = f"""You are an expert assistant trained on document information.
    you should answer hospital related questions.
    Use this context to answer the question:

    {formatted_context}

    Question: {query}

    Answer in detail using only the provided context:
    Answer with in a single paragraph:"""
    
    response = ollama.generate(
        model='deepseek-r1:1.5b',
        prompt=prompt,
        options={'temperature': 0.3, 'max_tokens': 2000}
    )
    return response['response']


# ----------------------
# Typing effect
# ----------------------
def typing_effect(text, delay=0.03):
    typed_text = ""
    placeholder = st.empty()

    for char in text:
        typed_text += char
        placeholder.markdown(f"**Answer:** {typed_text}")
        time.sleep(delay)


def get_tags():
    parent_folder = "uploaded_pdfs"  # change to your path
    if not os.path.isdir(parent_folder):
        return []
    folder_names = [
        name for name in os.listdir(parent_folder)
        if os.path.isdir(os.path.join(parent_folder, name))
    ]
    return folder_names


# ----------------------
# Cached model loaders
# ----------------------
# Streamlit reruns the whole script on every interaction (button click, text
# input, etc). Without caching, both the embedder and the Whisper model would
# be reloaded from disk every single time, which is slow and wasteful.

@st.cache_resource
def get_embedder(model_name="all-MiniLM-L6-v2"):
    return SentenceTransformer(model_name)


@st.cache_resource
def get_whisper_model(model_size="medium"):
    return whisper.load_model(model_size)


@st.cache_resource
def get_tts_pipeline(lang_code="a"):
    return KPipeline(lang_code=lang_code)


# ----------------------
# Voice transcription
# ----------------------
def transcribe_audio(audio_file) -> str:
    """
    Transcribe audio captured by st.audio_input (an in-memory WAV blob).
    Writes it to a temp file since whisper.transcribe expects a file path
    (or a numpy array — a temp file is the simplest, most robust option).
    """
    model = get_whisper_model()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_file.getvalue())
        tmp_path = tmp.name

    try:
        result = model.transcribe(tmp_path)
        return result["text"].strip()
    finally:
        os.remove(tmp_path)


KOKORO_SAMPLE_RATE = 24000

import re

def clean_text(text):
    # Keep letters, numbers, commas, periods, question marks,
    # exclamation marks, apostrophes, hyphens, and spaces.
    text = re.sub(r"[^a-zA-Z0-9.,!?'\-\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()



def text_to_speech(text: str, voice: str = "am_michael") -> bytes:
    """
    Convert text to speech with Kokoro and return a single in-memory WAV file
    (as bytes) ready for st.audio(). Kokoro yields audio in chunks (roughly
    one per sentence/clause) rather than one array for the whole input, so
    the chunks are concatenated into a single clip instead of writing them
    out as separate speech_0.wav, speech_1.wav, ... files.
    """
    pipeline = get_tts_pipeline()
    text =  clean_text(text)
    
    generator = pipeline(text, voice=voice)

    chunks = [audio for _, _, audio in generator]
    if not chunks:
        return b""

    full_audio = np.concatenate(chunks)

    buffer = io.BytesIO()
    sf.write(buffer, full_audio, KOKORO_SAMPLE_RATE, format="WAV")
    buffer.seek(0)
    return buffer.read()


# ----------------------
# Shared query pipeline (used by both text and voice input)
# ----------------------
def run_query(query: str, selected_tag: str):
    if not selected_tag:
        st.warning("⚠️ Please upload PDFs and select a tag first.")
        return

    index_path = os.path.join("vector_data", f"{selected_tag}_index", "vectors.index")
    docs_path = os.path.join("vector_data", f"{selected_tag}_index", "metadata.pkl")

    if not os.path.exists(index_path) or not os.path.exists(docs_path):
        st.error(f"No index found for tag '{selected_tag}'. Please (re)upload PDFs for this tag.")
        return

    with st.spinner("🤖 I am Thinking..."):
        index, documents, embedder = load_vector_store(index_path, docs_path)
        context = retrieve_context(query, embedder, index, documents)
        answer = generate_answer_with_ollama(query, context)
        answer = remove_think_tags(answer)

    typing_effect(answer)

    with st.spinner("🔊 Generating voice answer..."):
        try:
            audio_bytes = text_to_speech(answer)
        except Exception as exc:
            # TTS failing shouldn't hide the text answer that's already shown above.
            st.warning(f"Could not generate audio for this answer: {exc}")
            audio_bytes = b""

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")


# ----------------------
# Streamlit App
# ----------------------
st.title("📄 AI Chatbot")


st.header("📤 Upload PDFs")

tag_name = st.text_input(
    "Enter Tag Name",
    placeholder="hospital"
)

uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

if st.button("Upload & Create Index"):

    if tag_name == "":
        st.warning("Please enter a tag.")

    elif not uploaded_files:
        st.warning("Please upload PDF files.")

    else:

        folder = os.path.join(
            "uploaded_pdfs",
            tag_name
        )

        os.makedirs(folder, exist_ok=True)

        for file in uploaded_files:

            with open(
                os.path.join(folder, file.name),
                "wb"
            ) as f:

                f.write(file.getbuffer())

        with st.spinner("Creating Vector Store..."):

            created = process_uploaded_folder(tag_name)

        if created:
            st.success("✅ PDFs uploaded successfully!")
            st.success("✅ Vector Store Created!")
        else:
            st.error("No readable text found in the uploaded PDF(s).")

        st.rerun()


# ----------------------
# Chat interface
# ----------------------
tags = get_tags()
selected_tag = st.selectbox("Select a tag", tags) if tags else None

st.header("💬 Ask a Question")

query_label = f"Ask your question about {selected_tag}" if selected_tag else "Ask your question"

# if "userquestion" not in st.session_state:
#     st.session_state.userquestion = ""

#query = st.text_area(query_label, height=100)
query = st.text_area(
    query_label,
    key="userquestion",
    height=100
)

if st.button("Get Answer"):
    if not query:
        st.warning("⚠️ Please enter a question.")
    else:
        run_query(query, selected_tag)


st.header("🎙️ Or Ask by Voice")
st.caption("Click to record, speak your question, then click again to stop.")

audio_value = st.audio_input("Record your question")

if audio_value is not None:
    with st.spinner("Transcribing..."):
        transcribed_query = transcribe_audio(audio_value)

    if not transcribed_query:
        st.warning("⚠️ Couldn't make out any speech. Try recording again.")
    else:
        st.markdown(f"**Heard:** {transcribed_query}")

        # Auto-fill the text area
        #st.session_state.userquestion = transcribed_query

        if st.button("Get Answer (Voice)"):
            run_query(transcribed_query, selected_tag)

        # if st.button("Get Answer (Voice)"):
        #     run_query(transcribed_query, selected_tag)
