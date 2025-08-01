import streamlit as st
import tensorflow as tf
import numpy as np
from scipy.io import wavfile
import io

# --- Function to load and preprocess the audio ---
def preprocess_audio(audio_bytes):
    # Decode the WAV file
    sample_rate, audio_data = wavfile.read(io.BytesIO(audio_bytes))

    # Ensure audio is 16000Hz, if not, this part needs resampling (which is more complex)
    # For this simple app, we'll assume the uploaded audio is already 16kHz
    if sample_rate != 16000:
        st.error(f"Error: Audio must be 16kHz, but file is {sample_rate}Hz. Please resample your audio.")
        return None

    # Convert to float32
    audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max

    # If stereo, convert to mono
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    # Pad or trim to 16000 samples (1 second)
    if len(audio_data) > 16000:
        audio_data = audio_data[:16000]
    else:
        audio_data = np.pad(audio_data, (0, 16000 - len(audio_data)), 'constant')

    return audio_data

# --- Streamlit App ---

st.title("Lotha Speech-to-Text ASR")
st.write("Upload an audio file (.wav) of a Lotha word and see the model's prediction.")

# Load the exported TensorFlow model
# This assumes the 'saved' directory is in the same folder as app.py
try:
    model = tf.saved_model.load("saved")
except OSError:
    st.error("Model not found. Please make sure the 'saved' directory is in the same folder as this script.")
    st.stop()


uploaded_file = st.file_uploader("Choose a .wav file...", type="wav")

if uploaded_file is not None:
    # Read the audio file
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes, format='audio/wav')

    # Preprocess the audio
    waveform = preprocess_audio(audio_bytes)

    if waveform is not None:
        # Add a batch dimension and run inference
        waveform_tensor = tf.convert_to_tensor(waveform, dtype=tf.float32)
        waveform_tensor = tf.expand_dims(waveform_tensor, axis=0) # Create a batch of 1

        # Get the prediction from the model
        prediction = model(waveform_tensor)
        predicted_class = prediction['class_names'][0].numpy().decode('utf-8')

        st.success(f"Predicted Word: **{predicted_class}**")
