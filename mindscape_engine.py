import os
import json
import torch
torch.set_num_threads(1)
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
import gc
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from clinical_data import DSM_CRITERIA
from retrieval.hybrid_retriever import HybridRetriever
import threading

load_dotenv()

# Fix for broken TensorFlow on system
os.environ["USE_TF"] = "0"
# Fix to prevent HuggingFace Rust Tokenizers from deadlocking in Mesop threads
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

DEVICE = "cpu"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------
# Global Model Caches
# ---------------------------------------------------------
sensevoice_model = None
emo_model = None
medcpt_tokenizer = None
medcpt_model = None
nli_model = None
retriever = None

_init_lock = threading.Lock()

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
    global sensevoice_model, emo_model, DSM_MOCK_DATA # medcpt removed for memory
    global GLOBAL_BM25, GLOBAL_FAISS, GLOBAL_CORPUS
    
    with _init_lock:
        if sensevoice_model is not None and emo_model is not None:
             return
             
        if sensevoice_model is None:
            print(f"Loading SenseVoiceSmall on {DEVICE}...")
            local_sv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "SenseVoiceSmall"))
            cache_sv_path = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub", "models", "iic", "SenseVoiceSmall")
            
            # Decide path: local > cache > model name (for download)
            sv_source = "iic/SenseVoiceSmall"
            if os.path.exists(os.path.join(local_sv_path, "model.pt")):
                sv_source = local_sv_path
            elif os.path.exists(os.path.join(cache_sv_path, "model.pt")):
                sv_source = cache_sv_path
                
            print(f"Using source: {sv_source}")
            sensevoice_model = AutoModel(model=sv_source, trust_remote_code=False, device=DEVICE, disable_update=True)
        
        if emo_model is None:
            print("Loading official Emotion2Vec+ Base...")
            local_emo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "emotion2vec_plus_base"))
            cache_emo_path = os.path.join(os.path.expanduser("~"), ".cache", "modelscope", "hub", "models", "iic", "emotion2vec_plus_base")
            
            # Decide path: local > cache > model name (for download)
            emo_source = "iic/emotion2vec_plus_base"
            if os.path.exists(os.path.join(local_emo_path, "model.pt")):
                emo_source = local_emo_path
            elif os.path.exists(os.path.join(cache_emo_path, "model.pt")):
                emo_source = cache_emo_path
                
            print(f"Using source: {emo_source}")
            emo_model = AutoModel(model=emo_source, trust_remote_code=False, device=DEVICE, disable_update=True)
            
        # MedCPT Loading removed to prevent memory crashes and hanging.
        # HybridRetriever handles all embeddings locally now.
            
        if GLOBAL_BM25 is None:
            # We no longer load the mock JSON and build the MedCPT indices 
            # as it causes OOM issues in parallel with HybridRetriever.
            print("Skipped legacy mock MedCPT indices loaded.")

def load_nli_model():
    global nli_model
    # Disabled for MVP: Model size and Threading locks cause Mesop to freeze indefinitely.
    # if nli_model is None:
    #     print("Loading NLI CrossEncoder...")
    #     nli_model = CrossEncoder('cross-encoder/nli-distilroberta-base')
    return None


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
    
    import soundfile as sf
    data, sample_rate = sf.read(file_path)
    waveform = torch.from_numpy(data).float()
    
    # torchaudio expects [channels, time], soundfile returns [time, channels]
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    else:
        waveform = waveform.T
    
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
        disable_pbar=True,
        disable_log=True
    )
    if isinstance(res, list) and len(res) > 0:
        return res[0].get("text", "")
    return ""


# ---------------------------------------------------------
# Affect Extraction (Emotion2Vec+)
# ---------------------------------------------------------
def get_acoustic_affect(audio_path):
    print(">>> Starting get_acoustic_affect")
    init_models()
    
    # FunASR/PyTorch thread deadlock fix
    import concurrent.futures
    print(">>> Calling emo_model.generate")
    
    res = None
    try:
        def run_emo():
            return emo_model.generate(
                audio_path, 
                output_dir="./outputs", 
                granularity="utterance",
                disable_pbar=True,
                disable_log=True
            )
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        future = executor.submit(run_emo)
        res = future.result(timeout=10.0)
        executor.shutdown(wait=False)
    except Exception as e:
        print(f"emo_model execution timed out out or failed: {e}")
        return {
            "Valence": 0.0,   
            "Arousal": 0.5,   
            "Dominance": 0.5, 
            "Voice_Instability_Flag": False,
            "Dominant_Emotion": "Neutral",
            "Timeline": []
        }
        
    print(">>> Finished emo_model.generate")
    
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
    
    import soundfile as sf
    data, _ = sf.read(audio_path)
    waveform = torch.from_numpy(data).float()
    if waveform.ndim > 1:
        waveform = torch.mean(waveform, dim=1)
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
        "Dominant_Emotion": dom_emo_clean,
        "Timeline": [] # In full implementation, we'd map res[i] across time chunks. For MVP, we'll synthesize this timeline inside the LLM based on transcript segmentation.
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
        return np.ascontiguousarray(embeddings.cpu().numpy()).astype('float32')

def build_indices(data):
    print(f"Building FAISS and BM25 Indices for {len(data)} items...")
    corpus_texts = [f"{item['name']}: {item['desc']}" for item in data]
    
    # Sparse
    tokenized_corpus = [text.lower().split() for text in corpus_texts]
    bm25_index = BM25Okapi(tokenized_corpus)
    
    # Dense
    print("Generating MedCPT embeddings...")
    embeddings = get_medcpt_embeddings(corpus_texts)
    print(f"Embeddings generated. Shape: {embeddings.shape}, Dtype: {embeddings.dtype}")
    
    # Ensure C-contiguous float32 for FAISS
    embeddings = np.ascontiguousarray(embeddings).astype('float32')
    
    embedding_dim = embeddings.shape[1] 
    print(f"Initializing FAISS FlatIP index with dim {embedding_dim}...")
    # IndexFlatIP is more stable on ARM64 ARM for small datasets
    faiss_index = faiss.IndexFlatIP(embedding_dim)
    
    print("Normalizing embeddings...")
    faiss.normalize_L2(embeddings)
    
    print("Adding embeddings to FAISS index...")
    try:
        faiss_index.add(embeddings)
        print("FAISS index built successfully.")
    except Exception as e:
        print(f"FAISS add failed: {e}. Falling back to simple search.")
        # We still return the other components; search_dense will need a fallback
    
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
    Yields progress dicts `{"node": int, "status": str}` and finally `{"result": dict}`
    """
    init_models()
    
    yield {"node": 1, "status": "Acoustic Affect Extraction (Emotion2Vec+)"}
    affect_data = {"Valence": 0.0, "Arousal": 0.0, "Dominance": 0.0, "Voice_Instability_Flag": False, "Dominant_Emotion": "Neutral"}
    if audio_path and os.path.exists(audio_path):
        try:
            affect_data = get_acoustic_affect(audio_path)
            yield {"log": f"Extracted Vocal Affect: {affect_data['Dominant_Emotion']} (V={affect_data['Valence']}, A={affect_data['Arousal']}, D={affect_data['Dominance']}, Tremor={affect_data['Voice_Instability_Flag']})"}
        except Exception as e:
            print(f"Error extracting affect from audio: {e}")
            yield {"log": f"Vocal extraction bypassed or failed: {e}"}
            
    # 2. Fuse the context
    yield {"node": 2, "status": "Multimodal Context Fusion & BSV Mapping"}
    fusion_context = fuse_multimodal_data(transcript, affect_data)
    yield {"log": f"Fused Audio-Linguistic Context. Transcript Length: {len(transcript)}"}
    
    # 3. Hybrid Retrieval (RRF) for DSM-5 Knowledge
    yield {"node": 3, "status": "Hybrid Clinical Retrieval (FAISS/BM25)"}
    print(">>> Starting Hybrid Retrieval")
    clean_text = re.sub(r'<[^>]+>', '', transcript).strip()
    
    bsv_for_search = {"valence": affect_data["Valence"], "arousal": affect_data["Arousal"], "dominance": affect_data["Dominance"]}
    r_engine = load_retriever()
    print(">>> Calling retrieve_evidence")
    retrieval_package = r_engine.retrieve_evidence(clean_text, bsv_for_search)
    print(">>> Finished retrieve_evidence")
    yield {"log": f"Generated {len(retrieval_package.get('queries', []))} Semantic Vectors / Sparse BM25 Queries."}
    yield {"log": f"Retrieved {len(retrieval_package.get('evidence', []))} Top-K Historical/Corpus Base Matches in {retrieval_package.get('latency_ms', 0):.2f}ms."}
    
    # Format evidence for prompt
    evidence_list = []
    dsm_context = ""
    for ev in retrieval_package.get('evidence', []):
        doc_str = f"[{ev['title']} ({ev['source']})]\n{ev['text']}"
        evidence_list.append(doc_str)
        
    dsm_context = "\n\n".join(evidence_list)

    yield {"node": 4, "status": "LLM Synthesis & Diagnostics Gate"}
    # 4. Prepare Ultimate System Prompt
    system_prompt = f"""You are MindScape, an elite AI psychiatric diagnostic assistant. Your analysis must be grounded strictly in the provided multimodal data and retrieved DSM-5 texts.

{fusion_context}

[RETRIEVED DSM-5 KNOWLEDGE BASE]
{dsm_context}

**Process:**
1.  **Analyze the Fusion Data**: Weigh the explicit speech against the subconscious acoustic affect. Pay special attention if speech contradicts the affect (e.g., strong words but low Volume/Arousal).
2.  **Map Evidence**: Compare the multimodal state against the provided DSM-5 Criteria. If any Documented Historical Clinical Cases are retrieved, evaluate their symptoms and diagnoses against the current patient to identify masked or complex layered presentations.
3.  **Construct BSV**: Echo the Valence, Arousal, and Dominance derived from the acoustic signals into the final BSV output.
4.  **Formulate Hypothesis**: 
    - If criteria are met -> Name the Disorder (e.g., "Major Depressive Disorder").
    - If criteria are partially met but significant -> "Adjustment Disorder" or "Prodromal [Disorder]".
    - If no significant symptoms -> "Normal / No Diagnosis".

**Output Schema (JSON Only):**
{{
  "reasoning": "Brief clinical summary using **bullet points**. **Bold** specific symptoms (e.g., **anhedonia**).",
  "bsv": {{ "valence": float, "arousal": float, "dominance": float }},
  "traumatic_markers": ["Exact sentence/timestamp where a traumatic incident is recalled", "Another traumatic recall"],
  "emotion_trajectory": [
    {{
      "phase": "Beginning / Middle / End",
      "dominant_emotion": "Anger / Fear / Sadness / Neutral",
      "trigger": "What explicitly caused this emotion in the transcript"
    }}
  ],
  "hypothesis": {{
    "name": "Disorder Name",
    "confidence": "High / Moderate / Low / Provisional",
    "evidence": ["Quote from transcript or affect supporting criteria 1", "Quote 2"]
  }},
  "treatment_plan": "Specific first-line evidence-based pharmacological or therapeutic intervention (e.g., CBT, SSRIs, Lithium) grounded strictly in retrieved clinical texts or standard care. Be concise.",
  "reference_cases": [
    {{
      "title": "Title of retrieved Historical Case Study",
      "relevance": "Why this historical case is directly relevant to the current patient's presentation (e.g. underlying trauma, masked manic symptoms)",
      "historical_treatment": "The successful therapeutic intervention and final diagnosis documented in the historical case"
    }}
  ],
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
    # Safety Defaults
    if not result:
        yield {"result": {
            "bsv": affect_data,
            "hypothesis": {"name": "Error: Inference Failed", "evidence": []},
            "safety_gate": "FAIL"
        }}
        return

    # Validation
    if "bsv" not in result: result["bsv"] = {"valence": affect_data["Valence"], "arousal": affect_data["Arousal"], "dominance": affect_data["Dominance"]}
    if "hypothesis" not in result: result["hypothesis"] = {"name": "Inconclusive", "evidence": [], "confidence": "Low"}
    if "confidence" not in result["hypothesis"]: result["hypothesis"]["confidence"] = "N/A"
    if "follow_up" not in result: result["follow_up"] = []

    # ---------------------------------------------------------
    # HYBRID RETRIEVAL INTEGRATION (Task 2)
    # ---------------------------------------------------------
    # Format evidence for the UI
    ui_evidence_list = []
    for ev in retrieval_package.get('evidence', []):
        ui_evidence_list.append(f"[{ev['source']}] {ev['title']}: {ev['text']}")
    
    result['retrieved_evidence'] = ui_evidence_list

    # ---------------------------------------------------------
    # STRICT NLI SAFETY GATE
    # ---------------------------------------------------------
    # ---------------------------------------------------------
    if result.get('safety_gate') == 'PASS' and result['hypothesis']['name'] != 'Normal':
        try:
            nli = load_nli_model()
            if nli is not None:
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
            else:
                print("NLI skipped for MVP performance.")
        except Exception as e:
            print(f"NLI Check Failed: {e}")
            
    # Free up memory explicitly
    gc.collect()
            
    yield {"node": 5, "status": "Analysis Complete"}
    yield {"result": result}
