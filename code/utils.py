import os
import csv
import base64
import json
import hashlib
import time
from io import BytesIO
from PIL import Image
import pillow_avif
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Configure the Gemini client
api_key = os.environ.get("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

CACHE_FILE = ".openai_cache.json" # keep same cache file name to preserve cached images/data if any
LAST_CALL_TIME = 0.0

def resolve_path(path, base_dir="dataset"):
    """
    Resolves an image path from the CSV to the correct path on disk.
    """
    if not path:
        return ""
    # Normalize slashes
    path = path.replace("\\", "/")
    
    # If the path already has base_dir as a prefix, return it
    if path.startswith(f"{base_dir}/") or path.startswith(f"{base_dir}\\"):
        return path
        
    return os.path.join(base_dir, path)

def load_user_history(csv_path="dataset/user_history.csv"):
    """
    Loads user claim history from CSV and returns a dictionary indexed by user_id.
    """
    history = {}
    if not os.path.exists(csv_path):
        return history
        
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            history[row["user_id"]] = {
                "past_claim_count": int(row["past_claim_count"]),
                "accept_claim": int(row["accept_claim"]),
                "manual_review_claim": int(row["manual_review_claim"]),
                "rejected_claim": int(row["rejected_claim"]),
                "last_90_days_claim_count": int(row["last_90_days_claim_count"]),
                "history_flags": row["history_flags"],
                "history_summary": row["history_summary"]
            }
    return history

def load_evidence_requirements(csv_path="dataset/evidence_requirements.csv"):
    """
    Loads evidence requirements from CSV and returns a list of requirements.
    """
    reqs = []
    if not os.path.exists(csv_path):
        return reqs
        
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reqs.append({
                "requirement_id": row["requirement_id"],
                "claim_object": row["claim_object"],
                "applies_to": row["applies_to"],
                "minimum_image_evidence": row["minimum_image_evidence"]
            })
    return reqs

def encode_image_base64(image_path, max_size=1024):
    """
    Opens an image, resizes it if exceeds max_size, and returns a base64 encoded JPEG string.
    Returns None if the image file cannot be opened.
    """
    try:
        if not os.path.exists(image_path):
            # Try to resolve if it is relative to dataset
            resolved = resolve_path(image_path)
            if os.path.exists(resolved):
                image_path = resolved
            else:
                return None
                
        with Image.open(image_path) as img:
            # Convert to RGB to ensure jpeg compatibility
            if img.mode != 'RGB':
                img = img.convert('RGB')
                
            # Resize if exceeds max_size
            width, height = img.size
            if max(width, height) > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * (max_size / width))
                else:
                    new_height = max_size
                    new_width = int(width * (max_size / height))
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None

# Caching utility functions
def _load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cache: {e}")
    return {}

def _save_cache(cache):
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Error saving cache: {e}")

def get_cache_key(model, system_prompt, user_prompt, base64_images):
    """
    Computes a unique SHA-256 hash key for caching.
    """
    hasher = hashlib.sha256()
    hasher.update(model.encode("utf-8"))
    hasher.update(system_prompt.encode("utf-8"))
    hasher.update(user_prompt.encode("utf-8"))
    for img in base64_images:
        if img:
            hasher.update(img.encode("utf-8"))
    return hasher.hexdigest()

def call_gemini_vlm(model, system_prompt, user_prompt, base64_images, max_retries=5):
    """
    Calls the Google Gemini API with caching, rate limiting, and model failover.
    """
    global LAST_CALL_TIME
    cache = _load_cache()
    key = get_cache_key(model, system_prompt, user_prompt, base64_images)
    
    if key in cache:
        return cache[key]["response"], True # Return cached response and True for cache_hit
        
    # Check if configured
    if not api_key:
        current_api_key = os.environ.get("GEMINI_API_KEY")
        if current_api_key:
            genai.configure(api_key=current_api_key)
        else:
            raise ValueError("GEMINI_API_KEY is not configured in the environment.")
            
    # Prepare model pool starting with the preferred model
    pool = [model]
    for m in [
        "gemini-2.5-flash-lite",
        "gemini-3-flash-preview",
        "gemini-3.1-flash-lite",
        "gemini-flash-lite-latest",
        "gemini-2.5-flash"
    ]:
        if m not in pool:
            pool.append(m)
            
    contents = [user_prompt]
    for img_b64 in base64_images:
        if img_b64:
            contents.append({
                'mime_type': 'image/jpeg',
                'data': base64.b64decode(img_b64)
            })
            
    # Try models in the pool one by one
    for current_model in pool:
        print(f"Attempting VLM call using model: {current_model}...")
        retry_delay = 1.0
        for attempt in range(max_retries):
            try:
                # Enforce rate limit of 5 RPM (min 13 seconds between VLM calls)
                now = time.time()
                elapsed = now - LAST_CALL_TIME
                if elapsed < 13.0:
                    sleep_time = 13.0 - elapsed
                    print(f"Rate limiting: sleeping {sleep_time:.2f}s before API call...")
                    time.sleep(sleep_time)
                    
                # Instantiate model with system instructions
                model_instance = genai.GenerativeModel(
                    model_name=current_model,
                    system_instruction=system_prompt
                )
                
                # Call generation
                response = model_instance.generate_content(
                    contents=contents,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        temperature=0.0 # Deterministic where possible
                    )
                )
                
                response_text = response.text
                LAST_CALL_TIME = time.time()
                
                # Extract token counts
                try:
                    input_tokens = response.usage_metadata.prompt_token_count
                    output_tokens = response.usage_metadata.candidates_token_count
                except Exception:
                    input_tokens = 0
                    output_tokens = 0
                    
                # Parse response text as json
                parsed = json.loads(response_text)
                
                # Save to cache
                cache[key] = {
                    "response": parsed,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "timestamp": time.time(),
                    "actual_model_used": current_model
                }
                _save_cache(cache)
                
                return parsed, False
            except Exception as e:
                err_str = str(e)
                print(f"Gemini API call failed on attempt {attempt + 1} with model {current_model}: {e}")
                
                # If we hit a daily limit or project quota (e.g. GenerateRequestsPerDay), fail over to the next model in the pool!
                if "GenerateRequestsPerDay" in err_str or "limit: 20" in err_str or ("quota" in err_str.lower() and "day" in err_str.lower()):
                    print(f"Daily quota limit exceeded for model {current_model}. Failing over to next model in pool...")
                    break # Break out of the retry loop for this model to try the next model
                    
                if attempt == max_retries - 1:
                    print(f"All retries failed for model {current_model}.")
                    break # Try next model
                    
                # If rate limited (RPM), sleep longer
                if "429" in err_str or "ResourceExhausted" in err_str or "quota" in err_str.lower():
                    print("Rate limit error detected. Sleeping 30 seconds before retrying...")
                    time.sleep(30)
                else:
                    time.sleep(retry_delay)
                    retry_delay *= 2.0
                    
    # If we run out of models
    raise RuntimeError("All models in the pool failed to generate content.")
