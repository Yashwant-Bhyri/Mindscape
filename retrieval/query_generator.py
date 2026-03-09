import os
import json
from openai import OpenAI
import google.generativeai as genai

class QueryGenerator:
    def __init__(self):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.deepseek_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")

    def generate_queries(self, transcript, bsv):
        """
        Generate 3-5 retrieval queries based on transcript and BSV.
        """
        prompt = f"""
        Extract clinical symptoms and diagnostic search terms from the following transcript and behavioral state.
        
        TRANSCRIPT: {transcript}
        BEHAVIORAL STATE (BSV): Valence={bsv.get('valence', 'N/A')}, Arousal={bsv.get('arousal', 'N/A')}, Dominance={bsv.get('dominance', 'N/A')}
        
        Generate 3-5 short, keyword-rich search queries suitable for retrieving DSM criteria or psychiatric guidelines.
        Return ONLY a JSON list of strings.
        Example: ["manic episode symptoms", "DSM-5 bipolar criteria", "high arousal speech patterns"]
        """

        if self.gemini_key:
            try:
                genai.configure(api_key=self.gemini_key)
                model = genai.GenerativeModel('gemini-2.0-flash', 
                    generation_config={"response_mime_type": "application/json"})
                response = model.generate_content(prompt)
                return json.loads(response.text)
            except Exception as e:
                print(f"Query Gen Error (Gemini): {e}")

        if self.deepseek_key:
            try:
                base_url = os.getenv("OPENAI_BASE_URL")
                client = OpenAI(api_key=self.deepseek_key, base_url=base_url)
                
                model_id = "gpt-4o"
                if "openrouter" in (base_url or ""):
                    model_id = "deepseek/deepseek-chat"
                
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    max_tokens=256
                )
                content = response.choices[0].message.content
                # Sometimes LLMs wrap the list in a key
                data = json.loads(content)
                if isinstance(data, list): return data
                if isinstance(data, dict):
                    for val in data.values():
                        if isinstance(val, list): return val
                return [transcript[:50]] # Fallback
            except Exception as e:
                print(f"Query Gen Error (DeepSeek): {e}")

        # Fallback to simple keywords if no LLM
        return [transcript[:100]]

if __name__ == "__main__":
    # Test
    gen = QueryGenerator()
    queries = gen.generate_queries("I haven't slept in three days and I feel like I can fly.", {"valence": 0.8, "arousal": 0.9, "dominance": 0.9})
    print(f"Generated Queries: {queries}")
