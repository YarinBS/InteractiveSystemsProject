import streamlit as st
from streamlit_feedback import streamlit_feedback
from transformers import AutoTokenizer, AutoModel
from collections import Counter
import os

from text_generation_model import query_generation_model
from retrieval_model import get_top_texts
from doc_parser import create_chunks, docx_to_dict, txt_text, pdf_text
from constants import EMBEDDING_MODEL_PATH, EMBEDDING_MODEL_NAME, QUESTION_ANSWERING_MODEL_PATH, QUESTION_ANSWERING_MODEL_NAME, TOP_K, SIMILARITY_THRESHOLD, DEVICE

DATA_PATH = r'./Data'

st.title("Intelligent Interactive Systems Project")

if 'messages' not in st.session_state:
    st.session_state['messages'] = []

if 'feedback' not in st.session_state:
    st.session_state.feedback = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


def receive_feedback():
    try:
        if st.session_state.fb_k['score'] == "👍":
            st.toast("✔️ Positive Feedback received!")
        elif st.session_state.fb_k['score'] == "👎":
            st.toast("✔️ Negative Feedback received!")
    except TypeError:
        st.toast("No Feedback received.")


def get_feedback():
    with st.form('form'):
        st.write("[Optional] Provide a feedback...")
        streamlit_feedback(feedback_type="thumbs", align="center", key='fb_k')
        st.form_submit_button('Submit feedback', on_click=receive_feedback)


# Loading models and tokenizers
try:
    embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
    embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH, output_hidden_states=True, device_map=DEVICE)
except Exception as e:
    embedding_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME, output_hidden_states=True, device_map=DEVICE)

with st.sidebar:
    documents = st.file_uploader("Documents uploader", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    "Intelligent Interactive Systems 096235 - Technion"
    "               Winter 2024"
    " "
    "Made by Yarin Ben Shitrit and Orel Afriat"

if not documents:
    "👈 Welcome! Start by uploading your documents in the sidebar to the left! 👈"
    "⚠️ This is an AI-based 🤖 model, and it may provide incorrect/harmful responses. Use it with caution ⚠️"
else:
    "👇 Submit your questions in the text box below! 👇"


if documents: # and question:

    with st.chat_message("user"):
        question = st.chat_input(
            placeholder="Question goes here...",
            disabled=not documents
        )

    if question:
        with st.chat_message("user"):
            st.write(question)

        st.session_state.messages.append({"role": "user", "content": question})

        docs_with_chunks = {}

        for doc in documents:

            # if doc not in os.listdir(DATA_PATH):
            #     with open(f'{DATA_PATH}/{doc.name}', "wb") as f:
            #         f.write(doc.getbuffer())

            doc_extension = doc.name.split('.')[-1].lower()
            if doc_extension == 'docx':
                doc_text = docx_to_dict(f'{DATA_PATH}/{doc.name}')
            elif doc_extension == 'pdf':
                doc_text = pdf_text(f'{DATA_PATH}/{doc.name}')
            else:  # .txt file
                doc_text = txt_text(f'{DATA_PATH}/{doc.name}')

            chunks = create_chunks(doc_text, chunk_size=50, overlapping_size=10)
            docs_with_chunks[doc.name.split('/')[-1]] = chunks

        best_chunks = get_top_texts(question=question,
                                    docs=docs_with_chunks,
                                    top_k=TOP_K,
                                    similarity_threshold=SIMILARITY_THRESHOLD,
                                    model=embedding_model,
                                    tokenizer=embedding_tokenizer,
                                    device=DEVICE)

        if not best_chunks:  # Found nothing similar (that passes the similarity threshold)
            msg = "No relevant information was found in any document regarding the given question. Consider rewriting your question, or check if the desired answer appears in one of the existing documents."
            st.chat_message("assistant").write(msg)

            st.session_state.messages.append({"role": "assistant", "content": msg})

            get_feedback()

        else:

            context = str([chunk[chunk.find(':') + 2:] for chunk in best_chunks if not chunk.startswith('None')])
            try:
                model_answer = query_generation_model(question=question,
                                                      context=context,
                                                      model_name_or_path=QUESTION_ANSWERING_MODEL_PATH)
            except Exception as e:
                model_answer = query_generation_model(question=question,
                                                      context=context,
                                                      model_name_or_path=QUESTION_ANSWERING_MODEL_NAME)

            # Removing infrequent docs from the doc list of best_chunks
            retrieved_docs = [chunk[:chunk.find('___')] for chunk in best_chunks if not chunk.startswith('None')]
            retrieved_docs_counter = Counter(retrieved_docs)

            if len(retrieved_docs_counter) > 1:
                min_key = min(retrieved_docs_counter, key=retrieved_docs_counter.get)
                if retrieved_docs_counter[min_key] < TOP_K / 5:
                    retrieved_docs_counter.pop(min_key, None)

            relevant_docs = set([k[4:] for k in retrieved_docs_counter.keys()])
            model_answer = f'{model_answer}\n\nRelevant documents: {relevant_docs}'

            st.chat_message("assistant").write(model_answer)
            st.session_state.messages.append({"role": "assistant", "content": model_answer})

            get_feedback()
