import streamlit as st
from transformers import pipeline
import tempfile  # Geçici dosya yöntemi için
# Streamlit sayfa başlığı
st.set_page_config(page_title="Audio Transcription & Entity Extraction", layout="centered")

# Başlık ve Giriş Bilgileri
st.title("Audio Transcription & Named Entity Extraction")
st.write("**Developer:** [Adınız Soyadınız] | **Student ID:** [Öğrenci Numaranız]")
st.markdown("Bu uygulama WAV formatındaki ses dosyalarını yazıya döker ve metindeki kişileri (Persons), organizasyonları (Organizations) ve konumları (Locations) algılar.")

# Kullanıcıdan WAV ses dosyasını yüklemesini isteyen kısım
uploaded_file = st.file_uploader("Lütfen bir WAV dosyası yükleyin:", type=["wav"])

# Hugging Face'den Whisper Tiny Modelini yükleme
# ------------------------------
# Load Whisper Model
# ------------------------------
@st.cache_resource
def load_whisper_model():
    """
    Load the Whisper model for audio transcription.
    """
    # chunk_length_s=30 diyerek 30 sn'den uzun kayıtları parçalıyoruz.
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        chunk_length_s=30  # Uzun kayıtlar otomatik parçalanır.
    )
    return asr_pipeline

# ------------------------------
# Load NER Model
# ------------------------------
@st.cache_resource
def load_ner_model():
    """
    Load the Named Entity Recognition (NER) model pipeline.
    """
    ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
    return ner_pipeline

# ------------------------------
# Transcription Logic
# ------------------------------
def transcribe_audio(uploaded_file, asr_pipeline):
    """
    Transcribe audio into text using the Whisper model.
    Args:
        uploaded_file: Audio file uploaded by the user.
        asr_pipeline: The loaded Whisper pipeline.
    Returns:
        str: Transcribed text from the audio file.
    """
    # Geçici bir dosya oluşturup veriyi oraya yazıyoruz:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded_file.read())
        tmp.flush()  # Dosyayı diske yazmak için

        # Model bu geçici .wav dosyasını okuyarak transkripsiyon yapacak
        result = asr_pipeline(tmp.name)

    # Eğer Whisper dosyayı parçalara ayırdıysa (liste döner), hepsini birleştiriyoruz
    if isinstance(result, list):
        transcription_text = " ".join(segment["text"] for segment in result)
    else:
        # Kısa dosyalarda tek parça (dict) dönebilir
        transcription_text = result["text"]

    return transcription_text

# ------------------------------
# Entity Extraction
# ------------------------------
def extract_entities(text, ner_pipeline):
    """
    Extract entities from transcribed text using the NER model.
    """
    ner_results = ner_pipeline(text)

    orgs = set()
    locs = set()
    pers = set()

    for entity in ner_results:
        entity_group = entity["entity_group"]
        entity_word  = entity["word"]
        
        if entity_group == "ORG":
            orgs.add(entity_word)
        elif entity_group == "LOC":
            locs.add(entity_word)
        elif entity_group == "PER":
            pers.add(entity_word)

    return {
        "ORG": list(orgs),
        "LOC": list(locs),
        "PER": list(pers)
    }

# ------------------------------
# Main Streamlit Application
# ------------------------------
def main():
    st.title("Meeting Transcription and Entity Extraction")

    # Lütfen bu kısımları kendi bilgilerinizle değiştirin
    STUDENT_NAME = "Mustafa İhsan Yüce"
    STUDENT_ID   = "150210333"
    st.write(f"*{STUDENT_ID} - {STUDENT_NAME}*")

    # Model yüklemelerini gerçekleştirelim
    asr_pipeline = load_whisper_model()
    ner_pipeline = load_ner_model()

    # Uygulama içinde WAV formatlı dosya yüklenmesini sağlayalım
    uploaded_file = st.file_uploader("Lütfen WAV formatında bir ses dosyası yükleyiniz:", type=["wav"])

    if uploaded_file is not None:
        # Kullanıcının yüklediği ses dosyasını çalabilmek için
        st.audio(uploaded_file, format="audio/wav")

        # 1) Transcribe
        with st.spinner("Ses dosyası metne dönüştürülüyor (Whisper Tiny)..."):
            transcription_text = transcribe_audio(uploaded_file, asr_pipeline)

        st.subheader("Transcribed Text")
        st.write(transcription_text)

        # 2) NER Extraction
        with st.spinner("Metindeki varlıklar tespit ediliyor (NER modeli)..."):
            entities = extract_entities(transcription_text, ner_pipeline)

        st.subheader("Extracted Entities")
        # Organizasyonlar (ORG)
        st.write("*Organizations (ORG)*")
        if len(entities["ORG"]) > 0:
            for org in entities["ORG"]:
                st.write("- ", org)
        else:
            st.write("Hiçbir organizasyon tespit edilmedi.")

        # Lokasyonlar (LOC)
        st.write("*Locations (LOC)*")
        if len(entities["LOC"]) > 0:
            for loc in entities["LOC"]:
                st.write("- ", loc)
        else:
            st.write("Hiçbir lokasyon tespit edilmedi.")

        # Kişiler (PER)
        st.write("*Persons (PER)*")
        if len(entities["PER"]) > 0:
            for per in entities["PER"]:
                st.write("- ", per)
        else:
            st.write("Hiçbir kişi tespit edilmedi.")

if _name_ == "_main_":
    main()

