import mindscape_engine
import os

print("--- Starting Offline Pipeline Simulation ---")
test_audio = "retrieval/mock_audio/test.wav" # Ensure there is an audio file to test, or we'll synthesize a dummy tone.

# Create a temporary dummy audio file 
import numpy as np
import scipy.io.wavfile as wav
fs = 16000
duration = 3
recording = (np.sin(2 * np.pi * 440 * np.arange(fs * duration) / fs) * 10000).astype(np.int16)
dummy_file = "dummy_test.wav"
wav.write(dummy_file, fs, recording)

print("1. Generating Mock Speech Transcript with Paralinguistic Tags...")
mock_transcript = "I just feel so overwhelmed <|breath|> I haven't slept in days to be honest <|sad|>. I don't know what's wrong with me <|throat|>"

print("2. Extracting Mock Acoustic Affect...")
mock_affect = mindscape_engine.get_acoustic_affect(dummy_file)
# Override for demonstration since pure sine wave will be 'Neutral'
mock_affect["Dominant_Emotion"] = "Sad"
mock_affect["Valence"] = -0.75
mock_affect["Arousal"] = 0.60
mock_affect["Dominance"] = 0.20

print("3. Executing Multimodal Fusion...")
fusion_prompt = mindscape_engine.fuse_multimodal_data(mock_transcript, mock_affect)

print("\n==================================")
print("FINAL PIPELINE SYNTHESIS (Sent to Gemini LLM):")
print("==================================")
print(fusion_prompt)

os.remove(dummy_file)
