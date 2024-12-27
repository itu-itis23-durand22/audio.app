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
st.markdown("Upload a business meeting audio file to:/n 1.Transcribe the meeting audio into text. /n 2. Extract key entities such as Persons, Organisations, Dates, and Locations.")

# Kullanıcıdan WAV ses dosyasını yüklemesini isteyen kısım
uploaded_file = st.file_uploader("Upload an audio file(WAV format)", type=["wav"])

# Hugging Face'den Whisper Tiny Modelini yükleme
@st.cache_resource
def load_whisper_model():
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    return processor, model

# Hugging Face'den NER (Named Entity Recognition) Modelini yükleme
@st.cache_resource
def load_ner_model():
    return pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

# Geçici bir WAV dosyası oluşturma ve işleme
def process_audio_file(file):
    # Geçici dosya oluştur
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name
    return temp_file_path

# Ses dosyasını transkribe etme
def transcribe_audio(file_path, processor, model):
    with open(file_path, "rb") as f:
        audio = f.read()
    
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        generated_ids = model.generate(inputs["input_features"])
        transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

# NER sonuçlarını gruplama
def group_entities(ner_results):
    persons = set()
    organizations = set()
    locations = set()
    for entity in ner_results:
        if entity["entity_group"] == "PER":
            persons.add(entity["word"])
        elif entity["entity_group"] == "ORG":
            organizations.add(entity["word"])
        elif entity["entity_group"] == "LOC":
            locations.add(entity["word"])
    return persons, organizations, locations

# Eğer dosya yüklenmişse işlemler başlasın
if uploaded_file:
    st.audio(uploaded_file, format="audio/wav")
    st.write("### Ses Dosyası İşleniyor...")

    # Model Yükleme
    whisper_processor, whisper_model = load_whisper_model()
    ner_pipeline = load_ner_model()

    # 1. Adım: Ses dosyasını geçici olarak işleme
    audio_path = process_audio_file(uploaded_file)

    # 2. Adım: Ses dosyasını transkribe etme
    st.write("#### Adım 1: Ses Transkripsiyonu")
    transcription = transcribe_audio(audio_path, whisper_processor, whisper_model)
    st.text_area("Transkript", transcription, height=150)

    # 3. Adım: Transkribe edilmiş metinden varlık çıkarımı (NER)
    st.write("#### Adım 2: İsim Varlıklarının Çıkarılması")
    ner_results = ner_pipeline(transcription)
    persons, organizations, locations = group_entities(ner_results)

    # Varlıkları Gruplandır ve Göster
    st.write("#### Çıkarılan İsim Varlıkları")
    st.subheader("Persons (PER)")
    st.write(", ".join(persons) if persons else "Kişi bulunamadı.")
    st.subheader("Organizations (ORG)")
    st.write(", ".join(organizations) if organizations else "Organizasyon bulunamadı.")
    st.subheader("Locations (LOC)")
    st.write(", ".join(locations) if locations else "Konum bulunamadı.")

    # Geçici dosyayı sil
    os.remove(audio_path)
