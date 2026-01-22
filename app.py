# app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf
import tempfile
import os
from model import XVectorNet
import traceback
import base64
import time

# ---------------- CONFIG ----------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "speaker_xvector_model.pth"
EMB_PATH = "speaker_embeddings.npy"   # saved speaker DB
N_MELS = 64

st.set_page_config(page_title="Speaker Recognition", layout="centered")

# ---------------- UTILITIES ----------------
def load_model(num_speakers):
    model = XVectorNet(num_speakers=num_speakers)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def extract_logmel_from_array(y, sr, n_mels=N_MELS):
    win = int(0.025 * sr)
    hop = int(0.010 * sr)
    n_fft = 1
    while n_fft < win:
        n_fft *= 2
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop, win_length=win, power=2.0)
    logmel = librosa.power_to_db(mel, ref=np.max)
    # per-utterance normalization
    m = logmel.mean(axis=1, keepdims=True)
    s = logmel.std(axis=1, keepdims=True) + 1e-10
    logmel = (logmel - m) / s
    return logmel.astype(np.float32)

def extract_logmel(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    return extract_logmel_from_array(y, sr)

def cosine_sim(a, b):
    a = torch.tensor(a)
    b = torch.tensor(b)
    return F.cosine_similarity(a, b, dim=0).item()

def load_speaker_db():
    if not os.path.exists(EMB_PATH):
        return {}
    data = np.load(EMB_PATH, allow_pickle=True).item()
    return data

def save_speaker_db(db):
    np.save(EMB_PATH, db)

def normalize(vec):
    return vec / (np.linalg.norm(vec) + 1e-10)

# ---------------- INFERENCE ----------------
@st.cache_resource
def get_model(num_speakers):
    return load_model(num_speakers)

def recognize_from_file(audio_path, model, speaker_db):
    try:
        feat = extract_logmel(audio_path)
        feat_tensor = torch.tensor(feat).unsqueeze(0).to(DEVICE)  # (1, 64, T)
        _, emb = model(feat_tensor)
        emb = emb.detach().cpu().numpy()[0]

        best_speaker = None
        best_score = -1
        for spk, spk_emb in speaker_db.items():
            score = cosine_sim(emb, spk_emb)
            if score > best_score:
                best_score = score
                best_speaker = spk

        return best_speaker, float(best_score)
    except Exception as e:
        st.error("Error during recognition: " + str(e))
        traceback.print_exc()
        return None, 0.0

# ---------------- WEBRTC AUDIO PROCESSOR FOR MIC ----------------
class Recorder(AudioProcessorBase):
    """
    Collect audio frames and store to a temporary WAV file when user clicks 'Stop'.
    We'll expose a global var via st.session_state to get the saved path.
    """
    def __init__(self):
        self.frames = []

    def recv(self, frame):
        # frame: av.AudioFrame
        pcm = frame.to_ndarray()  # shape (channels, samples)
        # convert to mono if stereo
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=0)
        self.frames.append(pcm)
        return frame

    def save(self, sr=48000):
        if len(self.frames) == 0:
            return None
        audio = np.concatenate(self.frames)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(tmp.name, audio, sr)
        return tmp.name

# ---------------- UI ----------------
st.title("🎙️ Speaker Recognition — Streamlit App")
st.write("Upload audio, record from your browser (5-8s) or manage the speaker DB (admin only).")

menu = st.sidebar.radio("Menu", ["Recognize (Upload)", "Recognize (Mic)", "Admin (Protected)", "About / Exit"])

# Load database
speaker_db = load_speaker_db()
num_registered = len(speaker_db)

if menu == "Recognize (Upload)":
    st.header("Upload a .wav file for recognition")
    uploaded = st.file_uploader("Choose a .wav file", type=["wav"], accept_multiple_files=False)
    if uploaded is not None:
        # save uploaded file to temp path
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        tfile.write(uploaded.getvalue())
        tfile.flush()
        st.audio(tfile.name)
        if st.button("Recognize"):
            if len(speaker_db) == 0:
                st.warning("No registered speakers in DB. Please enroll via Admin.")
            else:
                model = get_model(num_speakers=max(1, len(speaker_db)))
                with st.spinner("Recognizing..."):
                    spk, score = recognize_from_file(tfile.name, model, speaker_db)
                st.success(f"Speaker: {spk}   —   Confidence (cosine): {score:.4f}")

elif menu == "Recognize (Mic)":
    st.header("Record microphone (5–8 seconds) and recognize")
    st.write("Click **Start** to begin capturing audio in your browser. When finished click **Stop**. (Works in modern browsers).")
    # control duration via UI
    duration = st.slider("Recording duration (seconds)", min_value=3, max_value=10, value=6, step=1)
    rec_container = st.empty()

    if "recorder" not in st.session_state:
        st.session_state.recorder = None
    if "saved_audio" not in st.session_state:
        st.session_state.saved_audio = None

    if st.button("Start Recording"):
        # start webrtc streamer; user will see mic popup
        st.session_state.recorder = webrtc_streamer(
            key="mic",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            media_stream_constraints={"audio": True, "video": False},
            audio_processor_factory=Recorder,
            async_processing=True
        )
        st.success("Recording started. Click Stop after your capture completes.")

    if st.session_state.get("recorder") is not None:
        if st.session_state.recorder.state.playing:
            if st.button("Stop Recording"):
                proc = st.session_state.recorder.audio_processor
                # try to save with standard SR 48000
                saved = proc.save(sr=48000)
                st.session_state.saved_audio = saved
                st.session_state.recorder.stop()
                st.success(f"Saved recording: {saved}")
                st.audio(saved)

    if st.session_state.get("saved_audio"):
        if st.button("Recognize recording"):
            if len(speaker_db) == 0:
                st.warning("No registered speakers. Enroll via Admin.")
            else:
                model = get_model(num_speakers=max(1, len(speaker_db)))
                with st.spinner("Recognizing..."):
                    spk, score = recognize_from_file(st.session_state.saved_audio, model, speaker_db)
                st.success(f"Speaker: {spk}   —   Confidence: {score:.4f}")

elif menu == "Admin (Protected)":
    st.header("Admin: Manage speaker database (password protected)")
    pwd = st.text_input("Enter admin password", type="password")
    if pwd != "Deal@08":
        st.warning("Enter admin password to proceed.")
    else:
        st.success("Admin access granted.")
        st.subheader("1) Enroll new speaker (upload multiple wavs)")
        with st.form("enroll_form"):
            spk_name = st.text_input("Speaker name (unique id)", value="")
            files = st.file_uploader("Upload one or more .wav files for this speaker", type=["wav"], accept_multiple_files=True)
            submit = st.form_submit_button("Enroll speaker")
            if submit:
                if spk_name.strip() == "" or len(files) == 0:
                    st.error("Provide a speaker name and at least one wav file.")
                else:
                    # load (or create) model with current DB size + 1
                    db = load_speaker_db()
                    try:
                        model = get_model(num_speakers=max(1, len(db)+1))
                    except Exception as e:
                        st.error("Model load failed: " + str(e))
                        model = None

                    embeddings = []
                    for f in files:
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                        tmp.write(f.getvalue())
                        tmp.flush()
                        try:
                            feat = extract_logmel(tmp.name)
                            feat_tensor = torch.tensor(feat).unsqueeze(0).to(DEVICE)
                            _, emb = model(feat_tensor)
                            emb = emb.detach().cpu().numpy()[0]
                            embeddings.append(emb)
                        except Exception as e:
                            st.error(f"Failed processing {f.name}: {e}")

                    if len(embeddings) > 0:
                        mean_emb = normalize(np.mean(embeddings, axis=0))
                        db[spk_name] = mean_emb
                        save_speaker_db(db)
                        speaker_db = db
                        st.success(f"Enrolled {spk_name} with {len(embeddings)} samples.")
                    else:
                        st.error("No embeddings created. Check files and try again.")

        st.subheader("2) Show / Delete registered speakers")
        db = load_speaker_db()
        if len(db) == 0:
            st.info("No speakers enrolled yet.")
        else:
            st.write("Registered speakers:")
            for spk in db.keys():
                st.write(" - " + spk)
            del_spk = st.text_input("Enter speaker name to delete (exact):", value="")
            if st.button("Delete speaker"):
                if del_spk in db:
                    db.pop(del_spk)
                    save_speaker_db(db)
                    speaker_db = db
                    st.success(f"Deleted {del_spk}")
                else:
                    st.error("Speaker not found.")

        st.subheader("3) Rebuild speaker DB from features/ (optional)")
        if st.button("Rebuild DB from features folder"):
            # create DB by averaging embeddings from features/ using the model
            from pathlib import Path
            feat_dir = Path("features")
            if not feat_dir.exists():
                st.error("features/ folder not found.")
            else:
                model = get_model(num_speakers=max(1, len(db)))
                new_db = {}
                for spk in sorted(os.listdir("features")):
                    spk_path = os.path.join("features", spk)
                    if not os.path.isdir(spk_path):
                        continue
                    embeddings = []
                    for f in os.listdir(spk_path):
                        if f.endswith(".npy"):
                            feat = np.load(os.path.join(spk_path, f))
                            feat_tensor = torch.tensor(feat).unsqueeze(0).to(DEVICE)
                            _, emb = model(feat_tensor)
                            emb = emb.detach().cpu().numpy()[0]
                            embeddings.append(emb)
                    if len(embeddings) > 0:
                        new_db[spk] = normalize(np.mean(embeddings, axis=0))
                        st.write(f"Processed {spk}: {len(embeddings)} samples")
                save_speaker_db(new_db)
                speaker_db = new_db
                st.success("Rebuilt speaker DB and saved to speaker_embeddings.npy")

elif menu == "About / Exit":
    st.header("About this app")
    st.write("""
    - Streamlit app for speaker recognition using X-Vector TDNN.
    - Upload audio or record from your browser for recognition.
    - Admin area is password protected (provided).
    - Built for your Speaker_project.
    """)
    st.write("To stop the app, press Ctrl+C in the terminal where you ran `streamlit run app.py`.")

# Footer
st.markdown("---")
st.write(f"Registered speakers: **{len(load_speaker_db())}**")
st.write("App running on device: " + DEVICE)
