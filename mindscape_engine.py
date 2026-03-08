import os
import json
import torch
import torchaudio
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import faiss
import re
from funasr import AutoModel
from transformers import AutoTokenizer, AutoModel as TransformersAutoModel
from rank_bm25 import BM25Okapi
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
<<<<<<< HEAD
from clinical_data import DSM_CRITERIA
from retrieval.hybrid_retriever import HybridRetriever

load_dotenv()

# Fix for broken TensorFlow on system
os.environ["USE_TF"] = "0"

# Initialize model variables
model = None
=======

load_dotenv()

DEVICE = "cpu"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# ---------------------------------------------------------
# Global Model Caches
# ---------------------------------------------------------
sensevoice_model = None
emo_model = None
medcpt_tokenizer = None
medcpt_model = None
>>>>>>> cdfbf838fa4c54089f78c2c5ef5884c7c2f2ae25
nli_model = None
retriever = None

def load_retriever():
    global retriever
    if retriever is None:
        print("Loading Hybrid Retriever...")
        retriever = HybridRetriever()
        retriever.initialize()
    return retriever

# Global Indices 
GLOBAL_BM25 = None
GLOBAL_FAISS = None
GLOBAL_CORPUS = None
DSM_MOCK_DATA = None


def init_models():
    """Lazy load all heavy models and build retrieval indices upon first use."""
    global sensevoice_model, emo_model, medcpt_tokenizer, medcpt_model, DSM_MOCK_DATA
    global GLOBAL_BM25, GLOBAL_FAISS, GLOBAL_CORPUS
    
    if sensevoice_model is None:
        print(f"Loading SenseVoiceSmall on {DEVICE} from local...")
        local_sv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "SenseVoiceSmall"))
        cache_sv_path = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub", "models", "iic", "SenseVoiceSmall")
        sv_path = local_sv_path
        if not os.path.exists(os.path.join(local_sv_path, "model.pt")) and os.path.exists(os.path.join(cache_sv_path, "model.pt")):
            sv_path = cache_sv_path
        bpe_path = os.path.join(sv_path, "chn_jpn_yue_eng_ko_spectok.bpe.model")
        if not os.path.exists(bpe_path):
            print(f"CRITICAL ERROR: BPE Model not found at {bpe_path}")
        else:
            print(f"BPE Model verified at {bpe_path}")
        sensevoice_model = AutoModel(model=sv_path, trust_remote_code=False, device=DEVICE, disable_update=True)
    
    if emo_model is None:
        print("Loading official Emotion2Vec+ Base from local...")
        local_emo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "emotion2vec_plus_base"))
        cache_emo_path = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub", "models", "iic", "emotion2vec_plus_base")
        emo_path = local_emo_path
        if not os.path.exists(os.path.join(local_emo_path, "model.pt")) and os.path.exists(os.path.join(cache_emo_path, "model.pt")):
            emo_path = cache_emo_path
        emo_model = AutoModel(model=emo_path, trust_remote_code=False, device=DEVICE, disable_update=True)
        
    if medcpt_model is None:
        print("Loading ncbi/MedCPT-Query-Encoder...")
        medcpt_tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        medcpt_model = TransformersAutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to(DEVICE)
        
    if GLOBAL_BM25 is None:
        # Load local DSM-5 JSON and build index
        DSM_DATA_PATH = r"E:\Mindscape\Mindscape\extracted_dsm5.json"
        if os.path.exists(DSM_DATA_PATH):
            with open(DSM_DATA_PATH, "r", encoding="utf-8") as f:
                DSM_MOCK_DATA = json.load(f)
            print(f"SUCCESS: Loaded {len(DSM_MOCK_DATA)} disorders from {DSM_DATA_PATH}.")
        else:
            print(f"ERROR: File '{DSM_DATA_PATH}' not found. Falling back to simple mock.")
            DSM_MOCK_DATA = [{"id": "MDD", "name": "Major Depressive Disorder", "desc": "Depressed mood and loss of interest in activities."}]
            
        GLOBAL_BM25, GLOBAL_FAISS, GLOBAL_CORPUS = build_indices(DSM_MOCK_DATA)


def load_nli_model():
    global nli_model
    if nli_model is None:
        print("Loading NLI CrossEncoder...")
        nli_model = CrossEncoder('cross-encoder/nli-distilroberta-base')
    return nli_model


# ---------------------------------------------------------
# Audio Recording & ASR
# ---------------------------------------------------------
def record_audio(duration=5, fs=16000):
    """ Record audio from the default microphone. """
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    print("Recording finished.")
    
    temp_dir = os.path.join(os.environ.get("TEMP", "/tmp"), "mindscape_uploads")
    os.makedirs(temp_dir, exist_ok=True)
    filename = f"live_recording_{int(os.times().elapsed)}.wav"
    filepath = os.path.join(temp_dir, filename)
    
    wav.write(filepath, fs, recording)
    return filepath

def transcribe_audio(file_path):
    """ Transcribe audio file using SenseVoice. Output includes paralinguistic tags. """
    init_models()
    
    waveform, sample_rate = torchaudio.load(file_path)
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        
    audio_data = waveform.squeeze().numpy()

    res = sensevoice_model.generate(
        input=audio_data,
        cache={},
        language="auto",
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,
    )
    if isinstance(res, list) and len(res) > 0:
        return res[0].get("text", "")
    return ""


# ---------------------------------------------------------
# Affect Extraction (Emotion2Vec+)
# ---------------------------------------------------------
def get_acoustic_affect(audio_path):
    init_models()
    
    res = emo_model.generate(audio_path, output_dir="./outputs", granularity="utterance")
    
    scores_list = res[0].get('scores', [0]*8)
    labels_list = res[0].get('labels', ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Other', 'Sad', 'Surprised'])
    
    emo_dict = {label: score for label, score in zip(labels_list, scores_list)}
    
    happy = float(emo_dict.get('Happy', 0.0))
    sad = float(emo_dict.get('Sad', 0.0))
    angry = float(emo_dict.get('Angry', 0.0))
    fear = float(emo_dict.get('Fearful', 0.0))
    neutral = float(emo_dict.get('Neutral', 0.0))
    
    # Mapping to V/A/D Space
    pseudo_valence = happy - sad - (angry * 0.5) - (fear * 0.5)
    pseudo_valence = max(-1.0, min(1.0, pseudo_valence)) 
    
    pseudo_arousal = (angry * 1.0) + (fear * 1.0) + (happy * 0.8) - (sad * 0.5) - (neutral * 0.5)
    pseudo_arousal = max(0.0, min(1.0, (pseudo_arousal + 1.0) / 2.0)) 
    
    pseudo_dominance = (angry * 1.0) + (happy * 0.8) - (fear * 0.8) - (sad * 0.8)
    pseudo_dominance = max(0.0, min(1.0, (pseudo_dominance + 1.0) / 2.0)) 
    
    waveform, _ = torchaudio.load(audio_path)
    rms = torch.sqrt(torch.mean(waveform**2))
    is_quiet = int(rms < 0.01)
    
    dom_emo = "Neutral"
    if emo_dict:
        dom_emo = max(emo_dict.items(), key=lambda x: x[1])[0]
        
    # Translate bilingual tags to english for llm readability if present
    dom_emo_clean = dom_emo.split('/')[-1] if '/' in dom_emo else dom_emo
    
    return {
        "Valence": round(pseudo_valence, 3),   
        "Arousal": round(pseudo_arousal, 3),   
        "Dominance": round(pseudo_dominance, 3), 
        "Voice_Instability_Flag": bool(is_quiet),
        "Dominant_Emotion": dom_emo_clean
    }


# ---------------------------------------------------------
# Retrieval Functions (Dense + Sparse)
# ---------------------------------------------------------
def get_medcpt_embeddings(texts):
    inputs = medcpt_tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = medcpt_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        return embeddings.cpu().numpy()

def build_indices(data):
    print("Building FAISS and BM25 Indices...")
    corpus_texts = [f"{item['name']}: {item['desc']}" for item in data]
    
    # Sparse
    tokenized_corpus = [text.lower().split() for text in corpus_texts]
    bm25_index = BM25Okapi(tokenized_corpus)
    
    # Dense
    embeddings = get_medcpt_embeddings(corpus_texts)
    embedding_dim = embeddings.shape[1] 
    faiss_index = faiss.IndexHNSWFlat(embedding_dim, 32)
    faiss.normalize_L2(embeddings)
    faiss_index.add(embeddings)
    
    return bm25_index, faiss_index, corpus_texts

def search_sparse(query, top_k=5):
    tokenized_query = query.lower().split()
    doc_scores = GLOBAL_BM25.get_scores(tokenized_query)
    top_indices = np.argsort(doc_scores)[::-1][:top_k]
    return [(idx, doc_scores[idx]) for idx in top_indices]

def search_dense(query, top_k=5):
    query_emb = get_medcpt_embeddings([query])
    faiss.normalize_L2(query_emb)
    distances, indices = GLOBAL_FAISS.search(query_emb, top_k)
    return indices[0].tolist()

def hybrid_search_rrf(query, data_list, corpus_texts, top_k=3, k_rrf=60):
    sparse_results = search_sparse(query, top_k=10)
    dense_indices = search_dense(query, top_k=10)
    
    rrf_scores = {i: 0.0 for i in range(len(data_list))}
    
    for rank, (doc_idx, score) in enumerate(sparse_results):
        if score > 0:
            rrf_scores[doc_idx] += 1.0 / (k_rrf + rank + 1)
            
    for rank, doc_idx in enumerate(dense_indices):
        if doc_idx != -1:
            rrf_scores[doc_idx] += 1.0 / (k_rrf + rank + 1)
        
    ranked_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    final_results = []
    for doc_idx, rrf_score in ranked_docs[:top_k]:
        if rrf_score > 0:
            final_results.append({
                "disorder": data_list[doc_idx]["name"],
                "criteria": corpus_texts[doc_idx],
                "rrf_score": round(rrf_score, 4)
            })
    return final_results


def fuse_multimodal_data(transcript, affect_scores):
    tags = re.findall(r'<[^>]+>', transcript)
    clean_text = re.sub(r'<[^>]+>', '', transcript).strip()
    
    return f"""
[MULTIMODAL FUSION DATA]

1. SPEECH CONTENT (ASR):
   "{clean_text}"

2. PARALINGUISTIC CUES (Discrete Events):
   Detected Tags: {', '.join(tags) if tags else 'None'}

3. ACOUSTIC AFFECT (Continuous Signals via Emotion2Vec+):
   - Dominant Emotion Predicted: {affect_scores['Dominant_Emotion']}
   - Valence (-1 to 1): {affect_scores['Valence']}
   - Arousal (0 to 1):  {affect_scores['Arousal']}
   - Dominance (0 to 1):{affect_scores['Dominance']}
   - Voice Instability (Tremor): {'DETECTED' if affect_scores['Voice_Instability_Flag'] else 'Normal'}
"""

# ---------------------------------------------------------
# Main Orchestrator (Diagnosis)
# ---------------------------------------------------------
def get_diagnosis(transcript, audio_path=None):
    """
    Combines Multimodal Fusion (Task 1) and Hybrid Retrieval (Task 2).
    Analyzes the transcript and acoustic vectors against DSM criteria using LLM.
    """
    init_models()
    
    # 1. Acoustic Affect Extraction 
    affect_data = {"Valence": 0.0, "Arousal": 0.0, "Dominance": 0.0, "Voice_Instability_Flag": False, "Dominant_Emotion": "Neutral"}
    if audio_path and os.path.exists(audio_path):
        try:
            affect_data = get_acoustic_affect(audio_path)
        except Exception as e:
            print(f"Error extracting affect from audio: {e}")
            
    # 2. Fuse the context
    fusion_context = fuse_multimodal_data(transcript, affect_data)
    
    # 3. Hybrid Retrieval (RRF) for DSM-5 Knowledge
    clean_text = re.sub(r'<[^>]+>', '', transcript).strip()
    search_query = clean_text if clean_text else "patient presents normally"
    retrieved_docs = hybrid_search_rrf(search_query, DSM_MOCK_DATA, GLOBAL_CORPUS, top_k=3)
    
    dsm_context = "\n".join([f"[{d['disorder']}]\n{d['criteria']}" for d in retrieved_docs])

    # 4. Prepare Ultimate System Prompt
    system_prompt = f"""You are MindScape, an elite AI psychiatric diagnostic assistant. Your analysis must be grounded strictly in the provided multimodal data and retrieved DSM-5 texts.

{fusion_context}

[RETRIEVED DSM-5 KNOWLEDGE BASE]
{dsm_context}

**Process:**
1.  **Analyze the Fusion Data**: Weigh the explicit speech against the subconscious acoustic affect. Pay special attention if speech contradicts the affect (e.g., strong words but low Volume/Arousal).
2.  **Map Evidence**: Compare the multimodal state against the provided DSM-5 Criteria.
3.  **Construct BSV**: Echo the Valence, Arousal, and Dominance derived from the acoustic signals into the final BSV output.
4.  **Formulate Hypothesis**: 
    - If criteria are met -> Name the Disorder (e.g., "Major Depressive Disorder").
    - If criteria are partially met but significant -> "Adjustment Disorder" or "Prodromal [Disorder]".
    - If no significant symptoms -> "Normal / No Diagnosis".

**Output Schema (JSON Only):**
{{
  "reasoning": "Brief clinical summary using **bullet points**. **Bold** specific symptoms (e.g., **anhedonia**).",
  "bsv": {{ "valence": float, "arousal": float, "dominance": float }},
  "hypothesis": {{
    "name": "Disorder Name",
    "confidence": "High / Moderate / Low / Provisional",
    "evidence": ["Quote from transcript or affect supporting criteria 1", "Quote 2"]
  }},
  "follow_up": ["Specific question to ask the patient to clarify criteria 1"],
  "safety_gate": "PASS"
}}

**Critical Rules:**
- Do not hallucinate symptoms.
- If the audio is short (under 5 words) and affect is Normal, default to "Normal" and safety_gate "PASS".
- If the transcript is empty, output safety_gate "FAIL".
"""

    gemini_key = os.getenv("GEMINI_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    result = None

    if gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel('gemini-2.0-flash', 
                generation_config={"response_mime_type": "application/json"})
            response = model.generate_content(system_prompt)
            result = json.loads(response.text)
        except Exception as e:
             print(f"Gemini Error: {e}")
             result = None

    # Fallback to Deepseek/OpenAI
    if not result and deepseek_key:
        base_url = os.getenv("OPENAI_BASE_URL")
        client = OpenAI(api_key=deepseek_key, base_url=base_url)
        try:
            model_id = "gpt-4o"
            if "deepseek" in (os.getenv("OPENAI_BASE_URL") or "") or "DeepSeek" in (os.getenv("DEEPSEEK_API_KEY") or ""):
                model_id = "deepseek-chat"
            elif "openrouter" in (os.getenv("OPENAI_BASE_URL") or ""):
                model_id = "deepseek/deepseek-chat"

            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "system", "content": system_prompt}],
                response_format={"type": "json_object"},
                max_tokens=4096
            )
            result = json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"LLM Error: {e}")

    # Safety Defaults
    if not result:
        return {
            "bsv": affect_data,
            "hypothesis": {"name": "Error: Inference Failed", "evidence": []},
            "safety_gate": "FAIL"
        }

    # Validation
    if "bsv" not in result: result["bsv"] = {"valence": affect_data["Valence"], "arousal": affect_data["Arousal"], "dominance": affect_data["Dominance"]}
    if "hypothesis" not in result: result["hypothesis"] = {"name": "Inconclusive", "evidence": [], "confidence": "Low"}
    if "confidence" not in result["hypothesis"]: result["hypothesis"]["confidence"] = "N/A"
    if "follow_up" not in result: result["follow_up"] = []

    # ---------------------------------------------------------
    # HYBRID RETRIEVAL INTEGRATION (Task 2)
    # ---------------------------------------------------------
    try:
        bsv_data = result.get('bsv', {})
        r_engine = load_retriever()
        retrieval_package = r_engine.retrieve_evidence(transcript, bsv_data)
        
        # Format evidence for the UI
        evidence_list = []
        for ev in retrieval_package.get('evidence', []):
            evidence_list.append(f"[{ev['source']}] {ev['title']}: {ev['text']}")
        
        # Inject retrieved evidence into the result
        result['retrieved_evidence'] = evidence_list
        
        if evidence_list:
            if 'evidence' not in result['hypothesis']:
                 result['hypothesis']['evidence'] = []
            # Optionally append grounded ones or keep as separate list
            
    except Exception as e:
        print(f"Retrieval Integration Error: {e}")

    # ---------------------------------------------------------
    # STRICT NLI SAFETY GATE
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    if result.get('safety_gate') == 'PASS' and result['hypothesis']['name'] != 'Normal':
        try:
            nli = load_nli_model()
            diagnosis = result['hypothesis']['name']
            evidence_list = result['hypothesis']['evidence']
            
            nli_scores = []
            for ev in evidence_list:
                pair = (f"The patient presented with: {ev}", f"This indicates {diagnosis}")
                score = nli.predict([pair])[0] 
                probs = torch.softmax(torch.tensor(score), dim=0)
                nli_scores.append(float(probs[1]))
            
            avg_nli = sum(nli_scores) / len(nli_scores) if nli_scores else 0.0
            print(f"NLI Verification Score: {avg_nli:.4f}")
            if avg_nli < 0.15:
                print(f"Safety Gate Warning: Low NLI Score ({avg_nli:.4f})")
        except Exception as e:
            print(f"NLI Check Failed: {e}")
            
    return result
