import streamlit as st
from transformers import pipeline
import numpy as np
from pydub import AudioSegment
import io
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ------------------------------
# Load Whisper Model
# ------------------------------
def load_whisper_model():
    """
    Load the Whisper model for audio transcription.
    """
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    return processor, model

# ------------------------------
# Load NER Model
# ------------------------------
def load_ner_model():
    """
    Load the Named Entity Recognition (NER) model pipeline.
    """
    return pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

# ------------------------------
# Transcription Logic
# ------------------------------
def transcribe_audio(uploaded_file, processor, model):
    """
    Transcribe audio into text using the Whisper model.
    Args:
        uploaded_file: Audio file uploaded by the user.
        processor: WhisperProcessor instance.
        model: WhisperForConditionalGeneration instance.
    Returns:
        str: Transcribed text from the audio file.
    """
    # Convert the uploaded file to audio format if needed
    audio = AudioSegment.from_file(uploaded_file)
    
    # Export audio to raw PCM format (wav)
    audio = audio.set_channels(1).set_frame_rate(16000)
    with io.BytesIO() as audio_buffer:
        audio.export(audio_buffer, format="wav")
        audio_buffer.seek(0)

        # Load the audio as numpy array and feed to Whisper model
        audio_array = np.frombuffer(audio_buffer.read(), dtype=np.int16)
        
        # Use the Whisper model to transcribe audio to text
        inputs = processor(audio_array, return_tensors="pt", sampling_rate=16000)
        predicted_ids = model.generate(inputs["input_values"])
        transcription = processor.decode(predicted_ids[0], skip_special_tokens=True)
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
    grouped_entities = {"ORG": [], "LOC": [], "PER": []}
    
    for entity in entities:
        if entity['entity_group'] == 'ORG':
            grouped_entities["ORG"].append(entity['word'])
        elif entity['entity_group'] == 'LOC':
            grouped_entities["LOC"].append(entity['word'])
        elif entity['entity_group'] == 'PER':
            grouped_entities["PER"].append(entity['word'])
    
    return grouped_entities

# ------------------------------
# Main Streamlit Application
# ------------------------------
def main():
    st.title("Meeting Transcription and Entity Extraction")

    # You must replace below
    STUDENT_NAME = "Yaren Yılmaz"
    STUDENT_ID = "yilmazy20@itu.edu.tr"
    st.write(f"**{STUDENT_ID} - {STUDENT_NAME}**")

    # Load the models
    whisper_processor, whisper_model = load_whisper_model()
    ner_pipeline = load_ner_model()

    # Upload audio file
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg", "flac"])
    if uploaded_file is not None:
        st.audio(uploaded_file, format="audio/wav")

        # Transcribe the audio
        transcription = transcribe_audio(uploaded_file, whisper_processor, whisper_model)
        st.write("### Transcription")
        st.write(transcription)

        # Extract entities from the transcription
        entities = extract_entities(transcription, ner_pipeline)
        st.write("### Extracted Entities")
        st.write(f"Organizations: {entities['ORG']}")
        st.write(f"Locations: {entities['LOC']}")
        st.write(f"Persons: {entities['PER']}")

if __name__ == "__main__":
    main()
