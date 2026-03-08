import os
os.environ["MODELSCOPE_CACHE"] = "xxx\.cache\modelscope\hub"
from funasr import AutoModel

print("1. Forcing clean pre-download of SenseVoiceSmall...")
try:
    sensevoice_model = AutoModel(model="iic/SenseVoiceSmall", trust_remote_code=True, disable_update=True)
    print("SenseVoiceSmall OK!")
except Exception as e:
    print(f"SenseVoice Error: {e}")

print("2. Forcing clean pre-download of Emotion2Vec+...")
try:
    emo_model = AutoModel(model="iic/emotion2vec_plus_base", trust_remote_code=True, disable_update=True)
    print("Emotion2Vec+ OK!")
except Exception as e:
    print(f"Emotion2Vec Error: {e}")
    
print("All Models securely cached.")
