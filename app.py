import streamlit as st
from transformers import pipeline
import os
import tempfile
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch

# Streamlit sayfa başlığı
st.set_page_config(page_title="Meeting Transcription & Entity Extraction", layout="centered")

# Başlık ve Giriş Bilgileri
st.title("Meeting Transcription and Entity Extraction")
st.write("150220333|Doğukan Duran")
st.markdown("Upload a business meeting audio file to: 1.Transcribe the meeting audio into text.  2. Extract key entities such as Persons, Organisations, Dates, and Locations.")

# Kullanıcıdan WAV ses dosyasını yüklemesini isteyen kısım
uploaded_file = st.file_uploader("Upload an audio file(WAV format)", type=["wav"])

import streamlit as st
from transformers import pipeline

# ------------------------------
# Load Whisper Model
# ------------------------------
def load_whisper_model():
    """
    Load the Whisper model for audio transcription using Hugging Face's transformers library.
    """
    # Using pipeline to load the Whisper model
    whisper = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    return whisper


# ------------------------------
# Load NER Model
# ------------------------------
def load_ner_model():
    """
    Load the Named Entity Recognition (NER) model pipeline.
    """
    ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
    return ner_pipeline


# ------------------------------
# Transcription Logic
# ------------------------------
def transcribe_audio(uploaded_file, whisper_model):
    """
    Transcribe audio into text using the Whisper model.
    Args:
        uploaded_file: Audio file uploaded by the user.
        whisper_model: The Whisper model for transcription.
    Returns:
        str: Transcribed text from the audio file.
    """
    # Use whisper to transcribe the uploaded audio
    transcription = whisper_model(uploaded_file)["text"]
    return transcription


# ------------------------------
# Entity Extraction
# ------------------------------
def extract_entities(text, ner_model):
    """
    Extract entities from transcribed text using the NER model.
    Args:
        text (str): Transcribed text.
        ner_model: NER pipeline loaded from Hugging Face.
    Returns:
        dict: Grouped entities (ORGs, LOCs, PERs).
    """
    entities = ner_model(text)
    # Group entities by category
    grouped_entities = {"ORGs": [], "LOCs": [], "PERs": []}
    for entity in entities:
        if entity['entity_group'] == "ORG":
            grouped_entities["ORGs"].append(entity['word'])
        elif entity['entity_group'] == "LOC":
            grouped_entities["LOCs"].append(entity['word'])
        elif entity['entity_group'] == "PER":
            grouped_entities["PERs"].append(entity['word'])
    # Remove duplicates by converting lists to sets
    grouped_entities = {key: list(set(value)) for key, value in grouped_entities.items()}
    return grouped_entities


# ------------------------------
# Main Streamlit Application
# ------------------------------
def main():
    st.title("Meeting Transcription and Entity Extraction")

    # You must replace below
    STUDENT_NAME = "Your Name"
    STUDENT_ID = "Your Student ID"
    st.write(f"**{STUDENT_ID} - {STUDENT_NAME}**")

    # File uploader
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav"])

    if uploaded_file is not None:
        # Load models
        whisper_model = load_whisper_model()
        ner_model = load_ner_model()

        # Transcribe audio
        transcription = transcribe_audio(uploaded_file, whisper_model)

        # Display transcription
        st.subheader("Transcription:")
        st.write(transcription)

        # Extract entities
        entities = extract_entities(transcription, ner_model)

        # Display extracted entities
        st.subheader("Extracted Entities:")
        for category, items in entities.items():
            st.write(f"**{category}:**")
            for item in items:
                st.write(f"- {item}")


if __name__ == "__main__":
    main()

