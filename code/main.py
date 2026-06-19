import os
import sys
import csv
from dotenv import load_dotenv
load_dotenv()

# Add current directory to sys.path to enable importing local utils and templates directly,
# avoiding collisions with the standard library 'code' module.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_user_history,
    load_evidence_requirements,
    encode_image_base64,
    call_gemini_vlm
)

from prompt_templates import SYSTEM_PROMPT, generate_user_prompt
from evaluation.main import post_process_predictions, FEW_SHOT_EXAMPLES

def main():
    # File Paths
    claims_csv = "dataset/claims.csv"
    user_history_csv = "dataset/user_history.csv"
    evidence_req_csv = "dataset/evidence_requirements.csv"
    
    # Target outputs
    output_root = "output.csv"
    output_dataset = "dataset/output.csv"
    
    # Load support files
    user_history = load_user_history(user_history_csv)
    evidence_requirements = load_evidence_requirements(evidence_req_csv)
    
    # Initialize Gemini client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable is not set.")
        return
        
    model = "gemini-2.5-flash"
    
    # Read test claims
    if not os.path.exists(claims_csv):
        print(f"Error: {claims_csv} not found.")
        return
        
    with open(claims_csv, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        claims = list(reader)
        
    print(f"Loaded {len(claims)} test claims to process.")
    
    # We will use Strategy A (Zero-Shot) because it achieved the best performance (80.0% accuracy) 
    # during evaluation on the sample dataset.
    sys_prompt = SYSTEM_PROMPT
    
    predictions = []
    
    for i, row in enumerate(claims):
        user_id = row["user_id"]
        image_paths = row["image_paths"]
        user_claim = row["user_claim"]
        claim_object = row["claim_object"]
        
        # Get image files
        img_paths_list = [p.strip() for p in image_paths.split(";") if p.strip()]
        img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_paths_list]
        
        # Encode images
        base64_images = []
        for path in img_paths_list:
            encoded = encode_image_base64(path)
            if encoded:
                base64_images.append(encoded)
                
        # Get user history context
        u_hist = user_history.get(user_id, {})
        u_summary = u_hist.get("history_summary", "No prior history")
        u_flags = u_hist.get("history_flags", "none")
        
        # Generate user prompt
        user_prompt = generate_user_prompt(
            claim_object=claim_object,
            user_claim=user_claim,
            user_history_summary=u_summary,
            user_history_flags=u_flags,
            evidence_requirements=evidence_requirements,
            image_ids=img_ids
        )
        
        try:
            pred_raw, was_cached = call_gemini_vlm(model, sys_prompt, user_prompt, base64_images)
            pred_processed = post_process_predictions(pred_raw, user_id, user_history)
            
            # Combine original input fields with processed predictions
            # The schema requires:
            # user_id, image_paths, user_claim, claim_object, evidence_standard_met,
            # evidence_standard_met_reason, risk_flags, issue_type, object_part,
            # claim_status, claim_status_justification, supporting_image_ids, valid_image, severity
            result_row = {
                "user_id": user_id,
                "image_paths": image_paths,
                "user_claim": user_claim,
                "claim_object": claim_object,
                **pred_processed
            }
            predictions.append(result_row)
            print(f"Processed test case {i+1}/{len(claims)} (User: {user_id}, Object: {claim_object}) -> Status: {result_row['claim_status']}")
        except Exception as e:
            print(f"Error processing test case {i+1}: {e}")
            
    # Write outputs
    headers = [
        "user_id", "image_paths", "user_claim", "claim_object", "evidence_standard_met",
        "evidence_standard_met_reason", "risk_flags", "issue_type", "object_part",
        "claim_status", "claim_status_justification", "supporting_image_ids", "valid_image", "severity"
    ]
    
    # Write to root output.csv
    with open(output_root, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(predictions)
        
    # Write to dataset/output.csv
    with open(output_dataset, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(predictions)
        
    print(f"\nProcessing complete! Predictions written to {output_root} and {output_dataset}")

if __name__ == "__main__":
    main()
