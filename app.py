import streamlit as st
from transformers import pipeline
import io

# ------------------------------
# Load Whisper Model
# ------------------------------
def load_whisper_model():
    """
    Load the Whisper model for audio transcription
    """
    whisper_model = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")
    return whisper_model


# ------------------------------
# Load NER Model
# ------------------------------
def load_ner_model():
    """
    Load the Named Entity Recognition (NER) model pipeline.
    """
    ner_model = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)
    return ner_model


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
    # Read the audio file into memory
    audio_bytes = uploaded_file.read()

    # Transcribe the audio using the Whisper model
    transcription = whisper_model(audio_bytes)["text"]
    return transcription


# ------------------------------
# Entity Extraction
# ------------------------------
def extract_entities(text, ner_pipeline):
    """
    Extract entities from transcribed text using the NER model.
    Args:
        text (str): Transcribed text.
        ner_pipeline: NER pipeline loaded from Hugging Face.
    Returns:
        dict: Grouped entities (ORGs, LOCs, PERs).
    """
    entities = ner_pipeline(text)
    grouped_entities = {"ORGs": [], "LOCs": [], "PERs": []}

    # Group entities by their entity type
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

    # Replace with your name and student ID
    STUDENT_NAME = "Doğukan Duran"
    STUDENT_ID = "150220333"
    st.write(f"**{STUDENT_ID} - {STUDENT_NAME}**")

    # File uploader for audio
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg", "flac"])

    if uploaded_file is not None:
        # Load models
        whisper_model = load_whisper_model()
        ner_model = load_ner_model()

        # Transcribe the audio
        try:
            transcription = transcribe_audio(uploaded_file, whisper_model)

            # Display transcription
            st.subheader("Transcription:")
            st.write(transcription)

            # Extract entities from the transcription
            entities = extract_entities(transcription, ner_model)

            # Display extracted entities
            st.subheader("Extracted Entities:")
            for category, items in entities.items():
                st.write(f"**{category}:**")
                for item in items:
                    st.write(f"- {item}")
        except Exception as e:
            st.error(f"An error occurred during transcription or entity extraction: {e}")


if __name__ == "__main__":
    main()
