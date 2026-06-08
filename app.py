import streamlit as st
import joblib
import re
import time
import os

import plotly.express as px
import plotly.graph_objects as go

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# =====================================================
# CONFIG
# =====================================================
st.set_page_config(
    page_title="Klasifikasi Judul Skripsi",
    page_icon="🎓",
    layout="wide"
)

# =====================================================
# LOAD MODEL (JOBLIB ONLY)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "model.joblib")
TFIDF_PATH = os.path.join(BASE_DIR, "tfidf.joblib")

try:
    model = joblib.load(MODEL_PATH)
    tfidf = joblib.load(TFIDF_PATH)

except Exception as e:
    st.error(f"Gagal load model / TF-IDF:\n{e}")
    st.stop()

# =====================================================
# NLP SETUP
# =====================================================
stopword = StopWordRemoverFactory().create_stop_word_remover()
stemmer = StemmerFactory().create_stemmer()

# =====================================================
# PREPROCESSING
# =====================================================
def preprocessing(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    text = stopword.remove(text)

    words = text.split()
    words = [w for w in words if len(w) > 1]

    text = " ".join(words)

    text = stemmer.stem(text)

    return text

# =====================================================
# UI
# =====================================================
st.title("🎓 Klasifikasi Judul Skripsi")
st.caption("TF-IDF + Machine Learning (JOBLIB VERSION)")

left, right = st.columns([1, 1])

with left:
    st.subheader("Input Judul Skripsi")

    judul = st.text_area(
        "Masukkan judul",
        height=180,
        placeholder="Contoh: Analisis Sentimen Twitter Menggunakan SVM"
    )

    btn = st.button("🚀 Prediksi")

# =====================================================
# PREDIKSI
# =====================================================
with right:
    st.subheader("Hasil Prediksi")

    if btn:

        if judul.strip() == "":
            st.warning("Judul tidak boleh kosong")

        else:

            with st.spinner("Memproses..."):
                time.sleep(1)

            clean = preprocessing(judul)

            try:
                X = tfidf.transform([clean])

                # ================================
                # HANDLE MODEL
                # ================================
                if hasattr(model, "predict_proba"):
                    prob = model.predict_proba(X)[0]

                    prob_stem = float(prob[0])
                    prob_nonstem = float(prob[1])

                else:
                    pred = model.predict(X)

                    # fallback (kalau model ANN / custom)
                    if hasattr(pred, "shape") and len(pred.shape) > 1:
                        pred = pred.flatten()

                    if len(pred) == 1:
                        prob_nonstem = float(pred[0])
                        prob_stem = 1 - prob_nonstem
                    else:
                        prob_stem = float(pred[0])
                        prob_nonstem = float(pred[1])

            except Exception as e:
                st.error(f"Error prediksi:\n{e}")
                st.stop()

            # =================================================
            # CLASSIFICATION
            # =================================================
            if prob_stem >= prob_nonstem:
                label = "STEM"
                confidence = prob_stem
            else:
                label = "NON STEM"
                confidence = prob_nonstem

            # =================================================
            # OUTPUT
            # =================================================
            if label == "STEM":
                st.success(f"Kategori: {label}")
            else:
                st.warning(f"Kategori: {label}")

            st.info(f"Confidence: {confidence*100:.2f}%")

            st.markdown("---")

            # =================================================
            # METRICS
            # =================================================
            c1, c2 = st.columns(2)

            c1.metric("STEM", f"{prob_stem*100:.2f}%")
            c2.metric("NON STEM", f"{prob_nonstem*100:.2f}%")

            # =================================================
            # PIE CHART
            # =================================================
            fig1 = px.pie(
                names=["STEM", "NON STEM"],
                values=[prob_stem*100, prob_nonstem*100],
                hole=0.6
            )

            st.plotly_chart(fig1, use_container_width=True)

            # =================================================
            # GAUGE
            # =================================================
            fig2 = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence*100,
                title={"text": "Confidence Score"},
                gauge={"axis": {"range": [0, 100]}}
            ))

            st.plotly_chart(fig2, use_container_width=True)

            # =================================================
            # PREPROCESS RESULT
            # =================================================
            st.subheader("Hasil Preprocessing")
            st.code(clean)

            # =================================================
            # SUMMARY
            # =================================================
            st.subheader("Kesimpulan")

            st.info(
                f"Judul diklasifikasikan sebagai **{label}** "
                f"dengan confidence **{confidence*100:.2f}%**"
            )