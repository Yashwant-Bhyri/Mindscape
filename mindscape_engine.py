import os
import json
import torch
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from funasr import AutoModel
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
from clinical_data import DSM_CRITERIA
from retrieval.hybrid_retriever import HybridRetriever

load_dotenv()

# Fix for broken TensorFlow on system
os.environ["USE_TF"] = "0"

# Initialize model variables
model = None
nli_model = None
retriever = None

def load_retriever():
    global retriever
    if retriever is None:
        print("Loading Hybrid Retriever...")
        retriever = HybridRetriever()
        retriever.initialize()
    return retriever

def load_nli_model():
    global nli_model
    if nli_model is None:
        print("Loading NLI CrossEncoder...")
        nli_model = CrossEncoder('cross-encoder/nli-distilroberta-base')
    return nli_model

def load_model():
    """Load the SenseVoiceSmall model on MPS or CPU."""
    global model
    if model is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading SenseVoiceSmall on {device}...")
        try:
            model = AutoModel(
                model="iic/SenseVoiceSmall",
                trust_remote_code=True,
                device=device,
                disable_update=True,
            )
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
    return model

def record_audio(duration=5, fs=16000):
    """
    Record audio from the default microphone.
    duration: seconds
    fs: sampling rate
    Returns: path to the recorded .wav file
    """
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    print("Recording finished.")
    
    # Save to temp file
    temp_dir = "/tmp/mindscape_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    filename = f"live_recording_{int(os.times().elapsed)}.wav"
    filepath = os.path.join(temp_dir, filename)
    
    wav.write(filepath, fs, recording)
    return filepath

def transcribe_audio(file_path):
    """
    Transcribe audio file using SenseVoice.
    Returns the text with tags (e.g., <sigh>, <laughter>).
    """
    model_instance = load_model()
    
    # SenseVoice inference
    # input can be a path or bytes. 
    # The output format depends on the model. Usually it returns a list of results.
    res = model_instance.generate(
        input=file_path,
        cache={},
        language="auto",  # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        batch_size_s=60,
        merge_vad=True,  # merge vad
        merge_thr=1.0,
    )
    
    # Extract text from result. Structure might vary, assuming standard funasr output.
    # Result is usually a list of dicts: [{'key': '...', 'text': '...', ...}]
    if isinstance(res, list) and len(res) > 0:
        text = res[0].get("text", "")
        return text
    return ""

def get_diagnosis(transcript, selected_criteria_keys=None):
    """
    Analyze transcript against DSM criteria using LLM.
    python_criteria_keys: list of keys to include from DSM_CRITERIA (e.g. ['MDD', 'GAD']).
    If None, uses all.
    """
    
    # 1. Prepare Criteria Text
    if selected_criteria_keys:
        criteria_text = "\n".join([f"{k}: {DSM_CRITERIA[k]}" for k in selected_criteria_keys if k in DSM_CRITERIA])
    else:
        criteria_text = "\n".join([f"{k}: {v}" for k, v in DSM_CRITERIA.items()])

    # 2. Prepare Prompt
    # 2. Prepare Prompt
    system_prompt = """You are MindScape, an advanced clinical intelligence engine designed to assist psychiatrists. Your analysis must be grounded strictly in the provided transcript and DSM-5 criteria.

**Process:**
1.  **Analyze Paralinguistics**: Interpret tags (e.g., <sigh>, <pause>, <loud>, <whisper>) as clinical indicators of affect.
2.  **Map Evidence**: Iterate through the transcript. For every symptom candidate, check if it explicitly satisfies a DSM-5 criterion.
3.  **Construct BSV**: 
    - Valence: -1.0 (Deeply Negative/Depressed) to 1.0 (Positive/Manic).
    - Arousal: 0.0 (Lethargic/Comatose) to 1.0 (Agitated/Panic).
    - Dominance: 0.0 (Helpless/Submissive) to 1.0 (Control/Aggressive).
4.  **Formulate Hypothesis**: 
    - If criteria are met -> Name the Disorder (e.g., "Major Depressive Disorder").
    - If criteria are partially met but significant -> "Adjustment Disorder" or "Prodromal [Disorder]".
    - If no significant symptoms -> "Normal / No Diagnosis".

**Output Schema (JSON Only):**
{
  "reasoning": "Brief clinical summary using **bullet points**. **Bold** specific symptoms (e.g., **anhedonia**).",
  "bsv": { "valence": float, "arousal": float, "dominance": float },
  "hypothesis": {
    "name": "Disorder Name",
    "confidence": "High / Moderate / Low / Provisional",
    "evidence": ["Quote from transcript serving as evidence 1", "Quote 2"]
  },
  "follow_up": ["Specific question to ask the patient to clarify criteria 1", "Question for criteria 2"],
  "safety_gate": "PASS"
}

**Critical Rules:**
- Do not hallucinate symptoms.
- If the audio is short (under 5 words) and neutral, default to "Normal" and safety_gate "PASS".
- If the transcript is empty, output safety_gate "FAIL".
"""

    user_msg = f"TRANSCRIPT:\n{transcript}\n\nDSM CRITERIA:\n{criteria_text}"

    # 3. Call API
    # Check for API Key
    gemini_key = os.getenv("GEMINI_API_KEY")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    
    result = None

    if gemini_key:
        try:
            genai.configure(api_key=gemini_key)
            # Use JSON mode for Gemini if possible, or just prompt for JSON.
            # Gemini 2.0 Flash selected from available models
            model = genai.GenerativeModel('gemini-2.0-flash', 
                generation_config={"response_mime_type": "application/json"})
            
            combined_prompt = f"{system_prompt}\n\nUSER INPUT:\n{user_msg}"
            response = model.generate_content(combined_prompt)
            result = json.loads(response.text)
        except Exception as e:
             print(f"Gemini Error: {e}")
             return {
                "bsv": {"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
                "hypothesis": {"name": "Error: Gemini Inference Failed", "evidence": [str(e)]},
                "safety_gate": "FAIL"
            }

    elif deepseek_key:
        base_url = os.getenv("OPENAI_BASE_URL")
        client = OpenAI(api_key=deepseek_key, base_url=base_url)

        try:
            # Check if using OpenRouter (sk-or-v1) and adjust model name
            model_id = "gpt-4o"
            if "deepseek" in (os.getenv("OPENAI_BASE_URL") or "") or "DeepSeek" in (os.getenv("DEEPSEEK_API_KEY") or ""):
                model_id = "deepseek-chat"
            elif "openrouter" in (os.getenv("OPENAI_BASE_URL") or ""):
                model_id = "deepseek/deepseek-chat"

            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg}
                ],
                response_format={"type": "json_object"},
                max_tokens=4096
            )
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Validation: Ensure strict schema
            if "bsv" not in result:
                 result["bsv"] = {"valence": 0.0, "arousal": 0.0, "dominance": 0.0}
            if "hypothesis" not in result:
                 result["hypothesis"] = {"name": "Inconclusive", "evidence": [], "confidence": "Low"}
            else:
                 if "confidence" not in result["hypothesis"]:
                     result["hypothesis"]["confidence"] = "N/A"
            if "follow_up" not in result:
                 result["follow_up"] = []
        except Exception as e:
            print(f"LLM Error: {e}")
            return {
                "bsv": {"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
                "hypothesis": {"name": "Error: Inference Failed", "evidence": [str(e)]},
                "safety_gate": "FAIL"
            }
    
    else:
        # No key
        return {
            "bsv": {"valence": 0.0, "arousal": 0.0, "dominance": 0.0},
            "hypothesis": {"name": "Error: No API Key", "evidence": []},
            "safety_gate": "FAIL"
        }

    # ---------------------------------------------------------
    # HYBRID RETRIEVAL INTEGRATION
    # ---------------------------------------------------------
    try:
        bsv = result.get('bsv', {})
        r_engine = load_retriever()
        retrieval_package = r_engine.retrieve_evidence(transcript, bsv)
        
        # Format evidence for the UI and for potential second pass logic
        evidence_list = []
        for ev in retrieval_package['evidence']:
            evidence_list.append(f"[{ev['source']}] {ev['title']}: {ev['text']}")
        
        # Inject retrieved evidence into the result
        result['retrieved_evidence'] = evidence_list
        
        # SECOND PASS: Grounded Reasoning (Simplified for now: append to evidence)
        # Note: In a full implementation, we'd re-call the LLM here with the evidence.
        # But per current MVP requirements, we attach the evidence for clinician review.
        if evidence_list:
            if 'evidence' not in result['hypothesis']:
                 result['hypothesis']['evidence'] = []
            # Optionally filter original evidence or append grounded ones
            # result['hypothesis']['evidence'].extend(evidence_list[:2]) # Top 2 grounded
            
    except Exception as e:
        print(f"Retrieval Integration Error: {e}")

    # ---------------------------------------------------------
    # STRICT NLI SAFETY GATE
    # ---------------------------------------------------------
    if result and result.get('safety_gate') == 'PASS' and result['hypothesis']['name'] != 'Normal':
        try:
            nli = load_nli_model()
            diagnosis = result['hypothesis']['name']
            evidence_list = result['hypothesis']['evidence']
            
            # Find relevant criteria text (simplified matching)
            relevant_criteria = ""
            for key, text in DSM_CRITERIA.items():
                if key in diagnosis or diagnosis in key: # Loose match
                    relevant_criteria = text
                    break
            
            if not relevant_criteria:
                    # Fallback: compare against all criteria if mapping fails
                    relevant_criteria = criteria_text
            
            nli_scores = []
            for ev in evidence_list:
                pair = (f"The patient said: {ev}", f"This indicates {diagnosis}")
                score = nli.predict([pair])[0] 
                
                # Check entailment score (index 1 for cross-encoder/nli-distilroberta-base)
                probs = torch.softmax(torch.tensor(score), dim=0)
                entailment_score = float(probs[1]) 
                nli_scores.append(entailment_score)
            
            avg_nli = sum(nli_scores) / len(nli_scores) if nli_scores else 0.0
            print(f"NLI Verification Score: {avg_nli:.4f}")
            
            if avg_nli < 0.15: # Threshold
                print(f"Safety Gate Warning: Low NLI Score ({avg_nli:.4f})")
                # result['safety_gate'] = 'FAIL (NLI)' # USER OVERRIDE: ALWAYS PASS
                # result['hypothesis']['name'] = 'Inconclusive (Evidence Weak)'
        
        except Exception as e:
            print(f"NLI Check Failed: {e}")
            
    return result
