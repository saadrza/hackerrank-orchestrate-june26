import os
import sys
import threading
import time
import csv
import json
from fastapi import FastAPI, BackgroundTasks, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Ensure the 'code/' directory is in python module path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils import (
    load_user_history,
    load_evidence_requirements,
    encode_image_base64,
    call_gemini_vlm,
    EXHAUSTED_MODELS,
    resolve_path
)
from prompt_templates import SYSTEM_PROMPT, generate_user_prompt, get_optimized_system_prompt
from evaluation.main import post_process_predictions

# Load environment variables
load_dotenv()

app = FastAPI(title="HackerRank Orchestrate VLM Dashboard")

# Mount images folder to serve visual evidence to the UI
images_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dataset", "images"))
if os.path.exists(images_dir):
    app.mount("/images", StaticFiles(directory=images_dir), name="images")

# Global Pipeline Run Status
PIPELINE_STATUS = {
    "running": False,
    "current": 0,
    "total": 0,
    "results": [],
    "model_used": "gemini-2.5-flash",
    "error": None
}

def clean_image_path(path):
    """
    Cleans paths like 'images/test/case_001/img_1.jpg' to render under /images/
    """
    path = path.replace("\\", "/")
    if path.startswith("dataset/images/"):
        path = path[len("dataset/images/"):]
    elif path.startswith("images/"):
        path = path[len("images/"):]
    return path

def run_predictions_task(preferred_model: str):
    global PIPELINE_STATUS
    try:
        claims_csv = "dataset/claims.csv"
        user_history_csv = "dataset/user_history.csv"
        evidence_req_csv = "dataset/evidence_requirements.csv"
        
        # Load support files
        user_history = load_user_history(user_history_csv)
        evidence_requirements = load_evidence_requirements(evidence_req_csv)
        
        if not os.path.exists(claims_csv):
            PIPELINE_STATUS["error"] = "claims.csv not found"
            PIPELINE_STATUS["running"] = False
            return
            
        with open(claims_csv, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            claims = list(reader)
            
        PIPELINE_STATUS["total"] = len(claims)
        PIPELINE_STATUS["current"] = 0
        PIPELINE_STATUS["results"] = []
        PIPELINE_STATUS["error"] = None
        
        for i, row in enumerate(claims):
            if not PIPELINE_STATUS["running"]:
                # Manual stop/interrupted
                break
                
            user_id = row["user_id"]
            image_paths = row["image_paths"]
            user_claim = row["user_claim"]
            claim_object = row["claim_object"]
            
            # Get image files
            img_paths_list = [p.strip() for p in image_paths.split(";") if p.strip()]
            img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_paths_list]
            
            # Clean image paths for frontend rendering
            cleaned_img_paths = [clean_image_path(p) for p in img_paths_list]
            
            # Encode images with Config 2 downscaling (max_size=512)
            base64_images = []
            for path in img_paths_list:
                encoded = encode_image_base64(path, max_size=512)
                if encoded:
                    base64_images.append(encoded)
                    
            # Get user history context
            u_hist = user_history.get(user_id, {})
            u_summary = u_hist.get("history_summary", "No prior history")
            u_flags = u_hist.get("history_flags", "none")
            
            # Build dynamic system prompt (Config 2 - Selective Context)
            sys_prompt = get_optimized_system_prompt(claim_object, shortened=False)
            
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
                # Call VLM using preferred model (will failover if needed)
                pred_raw, was_cached = call_gemini_vlm(preferred_model, sys_prompt, user_prompt, base64_images)
                pred_processed = post_process_predictions(pred_raw, user_id, user_history)
                
                result_row = {
                    "case_number": i + 1,
                    "user_id": user_id,
                    "image_paths": cleaned_img_paths,
                    "user_claim": user_claim,
                    "claim_object": claim_object,
                    "cached": was_cached,
                    **pred_processed
                }
                
                PIPELINE_STATUS["results"].append(result_row)
            except Exception as e:
                result_row = {
                    "case_number": i + 1,
                    "user_id": user_id,
                    "image_paths": cleaned_img_paths,
                    "user_claim": user_claim,
                    "claim_object": claim_object,
                    "cached": False,
                    "evidence_standard_met": "false",
                    "evidence_standard_met_reason": f"API Error: {str(e)}",
                    "risk_flags": "manual_review_required",
                    "issue_type": "unknown",
                    "object_part": "unknown",
                    "claim_status": "not_enough_information",
                    "claim_status_justification": f"VLM execution failed with error: {str(e)}",
                    "supporting_image_ids": "none",
                    "valid_image": "false",
                    "severity": "unknown"
                }
                PIPELINE_STATUS["results"].append(result_row)
                
            PIPELINE_STATUS["current"] = i + 1
            
        # Write outputs if full run completed
        if PIPELINE_STATUS["current"] == PIPELINE_STATUS["total"]:
            headers = [
                "user_id", "image_paths", "user_claim", "claim_object", "evidence_standard_met",
                "evidence_standard_met_reason", "risk_flags", "issue_type", "object_part",
                "claim_status", "claim_status_justification", "supporting_image_ids", "valid_image", "severity"
            ]
            
            # Translate cleaned images list back to string format
            csv_rows = []
            for r in PIPELINE_STATUS["results"]:
                restored_paths = ";".join([f"images/{p}" for p in r["image_paths"]])
                csv_rows.append({
                    "user_id": r["user_id"],
                    "image_paths": restored_paths,
                    "user_claim": r["user_claim"],
                    "claim_object": r["claim_object"],
                    "evidence_standard_met": r["evidence_standard_met"],
                    "evidence_standard_met_reason": r["evidence_standard_met_reason"],
                    "risk_flags": r["risk_flags"],
                    "issue_type": r["issue_type"],
                    "object_part": r["object_part"],
                    "claim_status": r["claim_status"],
                    "claim_status_justification": r["claim_status_justification"],
                    "supporting_image_ids": r["supporting_image_ids"],
                    "valid_image": r["valid_image"],
                    "severity": r["severity"]
                })
                
            with open("output.csv", mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(csv_rows)
                
            with open("dataset/output.csv", mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(csv_rows)
                
    except Exception as e:
        PIPELINE_STATUS["error"] = f"Fatal system error: {str(e)}"
    finally:
        PIPELINE_STATUS["running"] = False

@app.post("/api/run")
def start_pipeline(model: str = Query("gemini-2.5-flash")):
    global PIPELINE_STATUS
    if PIPELINE_STATUS["running"]:
        return {"status": "error", "message": "Pipeline is already running"}
        
    PIPELINE_STATUS["running"] = True
    PIPELINE_STATUS["current"] = 0
    PIPELINE_STATUS["total"] = 0
    PIPELINE_STATUS["results"] = []
    PIPELINE_STATUS["model_used"] = model
    PIPELINE_STATUS["error"] = None
    
    # Start task in background thread
    thread = threading.Thread(target=run_predictions_task, args=(model,))
    thread.daemon = True
    thread.start()
    
    return {"status": "success", "message": "Pipeline started"}

@app.post("/api/stop")
def stop_pipeline():
    global PIPELINE_STATUS
    if not PIPELINE_STATUS["running"]:
        return {"status": "error", "message": "Pipeline is not running"}
    PIPELINE_STATUS["running"] = False
    return {"status": "success", "message": "Pipeline stopped"}

@app.get("/api/status")
def get_status():
    return {
        "running": PIPELINE_STATUS["running"],
        "current": PIPELINE_STATUS["current"],
        "total": PIPELINE_STATUS["total"],
        "model_used": PIPELINE_STATUS["model_used"],
        "error": PIPELINE_STATUS["error"],
        "exhausted_models": list(EXHAUSTED_MODELS),
        "results": PIPELINE_STATUS["results"]
    }

@app.get("/api/models")
def get_models():
    models_list = [
        {"name": "gemini-2.5-flash", "label": "Gemini 2.5 Flash (Preferred)"},
        {"name": "gemini-2.5-flash-lite", "label": "Gemini 2.5 Flash Lite"},
        {"name": "gemini-3-flash-preview", "label": "Gemini 3 Flash (Preview)"},
        {"name": "gemini-3.1-flash-lite", "label": "Gemini 3.1 Flash Lite"},
        {"name": "gemini-flash-lite-latest", "label": "Gemini Flash Lite (Latest)"}
    ]
    
    results = []
    for m in models_list:
        status = "Exhausted" if m["name"] in EXHAUSTED_MODELS else "Available"
        results.append({
            "name": m["name"],
            "label": m["label"],
            "status": status
        })
    return results

@app.get("/", response_class=HTMLResponse)
def index():
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HackerRank Orchestrate Claims VLM Dashboard</title>
    <!-- Google Fonts -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0b0f19;
            --bg-secondary: #111827;
            --bg-card: #1f2937;
            --bg-glass: rgba(31, 41, 55, 0.65);
            --border-glass: rgba(255, 255, 255, 0.08);
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
            --accent-blue: #3b82f6;
            --accent-blue-glow: rgba(59, 130, 246, 0.35);
            --success: #10b981;
            --success-bg: rgba(16, 185, 129, 0.12);
            --danger: #ef4444;
            --danger-bg: rgba(239, 68, 68, 0.12);
            --warning: #f59e0b;
            --warning-bg: rgba(245, 158, 11, 0.12);
            --shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: radial-gradient(circle at 50% 0%, var(--bg-secondary) 0%, var(--bg-primary) 100%);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            padding: 2rem;
            overflow-x: hidden;
        }

        header {
            margin-bottom: 2rem;
            text-align: left;
            border-bottom: 1px solid var(--border-glass);
            padding-bottom: 1.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        h1 {
            font-family: 'Outfit', sans-serif;
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 50%, #1d4ed8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .subtitle {
            color: var(--text-secondary);
            font-size: 1rem;
            font-weight: 400;
        }

        /* Glassmorphism Panels */
        .glass-panel {
            background: var(--bg-glass);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid var(--border-glass);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: var(--shadow);
            margin-bottom: 2rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns: 1fr 2fr;
            gap: 2rem;
            align-items: start;
        }

        @media (max-width: 1024px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Form Controls */
        .form-group {
            margin-bottom: 1.25rem;
        }

        .form-label {
            display: block;
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-secondary);
            margin-bottom: 0.5rem;
        }

        select {
            width: 100%;
            background: rgba(17, 24, 39, 0.8);
            border: 1px solid var(--border-glass);
            color: var(--text-primary);
            padding: 0.75rem 1rem;
            border-radius: 8px;
            outline: none;
            font-size: 1rem;
            transition: all 0.3s ease;
        }

        select:focus {
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 3px var(--accent-blue-glow);
        }

        /* Buttons */
        .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            border: none;
            outline: none;
            text-align: center;
            width: 100%;
        }

        .btn-primary {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
            box-shadow: 0 4px 14px 0 rgba(37, 99, 235, 0.4);
        }

        .btn-primary:hover:not(:disabled) {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px 0 rgba(37, 99, 235, 0.6);
        }

        .btn-primary:disabled {
            background: #4b5563;
            color: #9ca3af;
            cursor: not-allowed;
            box-shadow: none;
        }

        .btn-danger {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
            box-shadow: 0 4px 14px 0 rgba(220, 38, 38, 0.4);
        }

        .btn-danger:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px 0 rgba(220, 38, 38, 0.6);
        }

        /* Model Availability Cards */
        .model-list {
            margin-top: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }

        .model-card {
            background: rgba(17, 24, 39, 0.4);
            border: 1px solid var(--border-glass);
            border-radius: 8px;
            padding: 0.75rem 1rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.875rem;
        }

        .model-name {
            font-weight: 500;
        }

        .model-status {
            display: inline-flex;
            align-items: center;
            gap: 0.35rem;
            font-weight: 600;
            font-size: 0.75rem;
            padding: 0.25rem 0.5rem;
            border-radius: 12px;
        }

        .status-available {
            color: var(--success);
            background: var(--success-bg);
        }

        .status-exhausted {
            color: var(--danger);
            background: var(--danger-bg);
        }

        /* Progress Bar */
        .progress-container {
            margin-top: 1.5rem;
            display: none;
        }

        .progress-meta {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.875rem;
            margin-bottom: 0.5rem;
            color: var(--text-secondary);
        }

        .progress-track {
            width: 100%;
            height: 10px;
            background: rgba(17, 24, 39, 0.8);
            border-radius: 5px;
            overflow: hidden;
            border: 1px solid var(--border-glass);
        }

        .progress-fill {
            width: 0%;
            height: 100%;
            background: linear-gradient(90deg, #3b82f6 0%, #60a5fa 100%);
            border-radius: 5px;
            transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            box-shadow: 0 0 8px #60a5fa;
        }

        /* Results Panel Table */
        .results-panel {
            min-height: 400px;
        }

        .results-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1rem;
            flex-wrap: wrap;
            gap: 1rem;
        }

        .filter-controls {
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }

        .filter-controls select {
            width: auto;
            padding: 0.5rem 1rem;
            font-size: 0.875rem;
        }

        .table-wrapper {
            width: 100%;
            overflow-x: auto;
            border-radius: 8px;
            border: 1px solid var(--border-glass);
        }

        table {
            width: 100%;
            border-collapse: collapse;
            text-align: left;
            font-size: 0.875rem;
        }

        th {
            background: rgba(17, 24, 39, 0.8);
            color: var(--text-secondary);
            font-weight: 600;
            padding: 1rem;
            border-bottom: 1px solid var(--border-glass);
        }

        td {
            padding: 1rem;
            border-bottom: 1px solid var(--border-glass);
            vertical-align: middle;
            background: rgba(31, 41, 55, 0.2);
            color: var(--text-primary);
        }

        tr:last-child td {
            border-bottom: none;
        }

        tr:hover td {
            background: rgba(31, 41, 55, 0.4);
        }

        /* Badges */
        .badge {
            display: inline-flex;
            align-items: center;
            font-weight: 600;
            font-size: 0.75rem;
            padding: 0.25rem 0.6rem;
            border-radius: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .badge-supported {
            color: var(--success);
            background: var(--success-bg);
            border: 1px solid rgba(16, 185, 129, 0.25);
        }

        .badge-contradicted {
            color: var(--danger);
            background: var(--danger-bg);
            border: 1px solid rgba(239, 68, 68, 0.25);
        }

        .badge-not_enough_information {
            color: var(--warning);
            background: var(--warning-bg);
            border: 1px solid rgba(245, 158, 11, 0.25);
        }

        .badge-cached {
            color: #60a5fa;
            background: rgba(96, 165, 250, 0.12);
            border: 1px solid rgba(96, 165, 250, 0.25);
            font-size: 0.65rem;
            margin-left: 0.5rem;
        }

        .badge-object {
            color: var(--text-secondary);
            background: rgba(156, 163, 175, 0.15);
            border: 1px solid rgba(156, 163, 175, 0.25);
        }

        /* Thumbnails */
        .thumbnail-gallery {
            display: flex;
            gap: 0.5rem;
        }

        .thumbnail {
            width: 50px;
            height: 50px;
            border-radius: 4px;
            object-fit: cover;
            border: 1px solid var(--border-glass);
            cursor: pointer;
            transition: transform 0.2s ease;
        }

        .thumbnail:hover {
            transform: scale(1.15);
            border-color: var(--accent-blue);
        }

        /* Modal Side Drawer */
        .drawer {
            position: fixed;
            top: 0;
            right: -550px;
            width: 500px;
            height: 100vh;
            background: #111827;
            border-left: 1px solid var(--border-glass);
            box-shadow: -10px 0 30px rgba(0,0,0,0.5);
            z-index: 1000;
            transition: right 0.4s cubic-bezier(0.4, 0, 0.2, 1);
            padding: 2rem;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
        }

        .drawer.open {
            right: 0;
        }

        .drawer-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-glass);
            padding-bottom: 1rem;
        }

        .drawer-title {
            font-family: 'Outfit', sans-serif;
            font-size: 1.5rem;
            font-weight: 600;
        }

        .drawer-close {
            background: none;
            border: none;
            color: var(--text-secondary);
            font-size: 1.5rem;
            cursor: pointer;
        }

        .drawer-close:hover {
            color: var(--text-primary);
        }

        .drawer-section-title {
            font-size: 0.875rem;
            font-weight: 600;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.5rem;
        }

        .drawer-chat-box {
            background: rgba(17, 24, 39, 0.5);
            border-radius: 8px;
            padding: 1rem;
            font-size: 0.875rem;
            max-height: 200px;
            overflow-y: auto;
            border: 1px solid var(--border-glass);
            white-space: pre-wrap;
        }

        .drawer-images {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 1rem;
        }

        .drawer-img-container {
            position: relative;
            aspect-ratio: 1;
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border-glass);
        }

        .drawer-img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            cursor: zoom-in;
            transition: transform 0.3s ease;
        }

        .drawer-img:hover {
            transform: scale(1.1);
        }

        .drawer-label-badge {
            position: absolute;
            bottom: 5px;
            left: 5px;
            background: rgba(0,0,0,0.7);
            color: white;
            font-size: 0.65rem;
            padding: 2px 6px;
            border-radius: 4px;
        }

        .drawer-justification {
            background: rgba(59, 130, 246, 0.05);
            border-left: 3px solid var(--accent-blue);
            padding: 1rem;
            border-radius: 0 8px 8px 0;
            font-size: 0.9rem;
            line-height: 1.5;
        }

        .drawer-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
        }

        .drawer-meta-item {
            background: rgba(17, 24, 39, 0.3);
            padding: 0.75rem;
            border-radius: 6px;
            border: 1px solid var(--border-glass);
        }

        .drawer-meta-value {
            font-weight: 600;
            margin-top: 0.25rem;
        }

        .overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background: rgba(0,0,0,0.5);
            backdrop-filter: blur(4px);
            z-index: 999;
            display: none;
        }

        .overlay.open {
            display: block;
        }

        /* Empty state styling */
        .empty-state {
            text-align: center;
            padding: 4rem 2rem;
            color: var(--text-secondary);
        }

        .empty-state-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }

        .action-row {
            display: flex;
            gap: 1rem;
            margin-top: 1rem;
        }
    </style>
</head>
<body>

    <header>
        <div>
            <h1>HackerRank Orchestrate Claims VLM Dashboard</h1>
            <div class="subtitle">Multi-Modal Damage Claims Verification Pipeline</div>
        </div>
        <div>
            <span class="model-status status-available" style="font-size: 0.9rem; padding: 0.5rem 1rem;">
                <span style="display:inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--success);"></span>
                FastAPI Dashboard Connected
            </span>
        </div>
    </header>

    <div class="dashboard-grid">
        
        <!-- Controls Sidebar -->
        <div class="glass-panel">
            <h2 class="drawer-title" style="margin-bottom: 1.5rem;">Pipeline Controls</h2>
            
            <div class="form-group">
                <label class="form-label" for="modelSelect">Select Google Gemini Model</label>
                <select id="modelSelect">
                    <!-- Loaded dynamically -->
                </select>
            </div>

            <div class="action-row">
                <button id="runBtn" class="btn btn-primary" onclick="startPipeline()">Run Verification Pipeline</button>
                <button id="stopBtn" class="btn btn-danger" style="display:none;" onclick="stopPipeline()">Cancel Run</button>
            </div>

            <div class="progress-container" id="progressContainer">
                <div class="progress-meta">
                    <span id="progressLabel">Initializing...</span>
                    <span id="progressPercent">0%</span>
                </div>
                <div class="progress-track">
                    <div class="progress-fill" id="progressFill"></div>
                </div>
                <div style="font-size: 0.75rem; color: var(--text-secondary); margin-top: 0.5rem;" id="progressSubtext">
                    Processing sequential VLM requests spaced at 13 seconds (5 RPM limit)...
                </div>
            </div>

            <h3 class="form-label" style="margin-top: 2rem; margin-bottom: 0.75rem;">Model Availability (Daily Quota)</h3>
            <div class="model-list" id="modelList">
                <!-- Loaded dynamically -->
            </div>
        </div>

        <!-- Results Table -->
        <div class="glass-panel results-panel">
            <div class="results-header">
                <h2 class="drawer-title">Claims Evaluation Results</h2>
                <div class="filter-controls">
                    <select id="filterObject" onchange="applyFilters()">
                        <option value="">All Objects</option>
                        <option value="car">Cars</option>
                        <option value="laptop">Laptops</option>
                        <option value="package">Packages</option>
                    </select>
                    <select id="filterStatus" onchange="applyFilters()">
                        <option value="">All Decisions</option>
                        <option value="supported">Supported</option>
                        <option value="contradicted">Contradicted</option>
                        <option value="not_enough_information">Not Enough Info</option>
                    </select>
                </div>
            </div>

            <div class="table-wrapper">
                <table id="resultsTable">
                    <thead>
                        <tr>
                            <th>Case</th>
                            <th>User ID</th>
                            <th>Object</th>
                            <th>Decision Status</th>
                            <th>Severity</th>
                            <th>Submitted Images</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody id="resultsBody">
                        <!-- Loaded dynamically -->
                        <tr>
                            <td colspan="7" class="empty-state">
                                <div class="empty-state-icon">🔬</div>
                                <div>No claims processed yet. Configure options and click Run to execute pipeline.</div>
                            </td>
                        </tr>
                    </tbody>
                </table>
            </div>
        </div>

    </div>

    <!-- Side Drawer Details Modal -->
    <div class="overlay" id="overlay" onclick="closeDrawer()"></div>
    <div class="drawer" id="drawer">
        <div class="drawer-header">
            <h3 class="drawer-title" id="drawerCaseTitle">Case Details</h3>
            <button class="drawer-close" onclick="closeDrawer()">&times;</button>
        </div>

        <div>
            <h4 class="drawer-section-title">Visual Evidence Submitted</h4>
            <div class="drawer-images" id="drawerImages">
                <!-- Loaded dynamically -->
            </div>
        </div>

        <div>
            <h4 class="drawer-section-title">Claim Conversation Transcript</h4>
            <div class="drawer-chat-box" id="drawerChat">
                <!-- Loaded dynamically -->
            </div>
        </div>

        <div>
            <h4 class="drawer-section-title">VLM Verification Justification</h4>
            <div class="drawer-justification" id="drawerJustification">
                <!-- Loaded dynamically -->
            </div>
        </div>

        <div class="drawer-grid">
            <div class="drawer-meta-item">
                <div class="drawer-section-title" style="margin-bottom: 0px;">Object / Part</div>
                <div class="drawer-meta-value" id="drawerPart">Car / Front Bumper</div>
            </div>
            <div class="drawer-meta-item">
                <div class="drawer-section-title" style="margin-bottom: 0px;">Severity</div>
                <div class="drawer-meta-value" id="drawerSeverity">Medium</div>
            </div>
            <div class="drawer-meta-item">
                <div class="drawer-section-title" style="margin-bottom: 0px;">Valid Image Set</div>
                <div class="drawer-meta-value" id="drawerValidImage">True</div>
            </div>
            <div class="drawer-meta-item">
                <div class="drawer-section-title" style="margin-bottom: 0px;">Standard Met</div>
                <div class="drawer-meta-value" id="drawerStandardMet">True</div>
            </div>
        </div>

        <div>
            <h4 class="drawer-section-title">Risk Flags Raised</h4>
            <div id="drawerRiskFlags" style="display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.25rem;">
                <!-- Loaded dynamically -->
            </div>
        </div>
    </div>

    <script>
        let allResults = [];
        let pollingInterval = null;

        // Fetch models availability status
        async function fetchModels() {
            try {
                const response = await fetch('/api/models');
                const models = await response.json();
                const selectEl = document.getElementById('modelSelect');
                const listEl = document.getElementById('modelList');
                
                // Save current selected if any
                const selectedVal = selectEl.value;
                
                selectEl.innerHTML = '';
                listEl.innerHTML = '';
                
                models.forEach(m => {
                    // Dropdown option
                    const opt = document.createElement('option');
                    opt.value = m.name;
                    opt.textContent = m.label + (m.status === 'Exhausted' ? ' [Exhausted]' : '');
                    selectEl.appendChild(opt);
                    
                    // Card
                    const statusClass = m.status === 'Available' ? 'status-available' : 'status-exhausted';
                    const card = document.createElement('div');
                    card.className = 'model-card';
                    card.innerHTML = `
                        <span class="model-name">${m.label}</span>
                        <span class="model-status ${statusClass}">
                            <span style="display:inline-block; width: 6px; height: 6px; border-radius: 50%; background: currentColor;"></span>
                            ${m.status}
                        </span>
                    `;
                    listEl.appendChild(card);
                });
                
                if (selectedVal && selectEl.querySelector(`option[value="${selectedVal}"]`)) {
                    selectEl.value = selectedVal;
                }
            } catch (err) {
                console.error("Error fetching models:", err);
            }
        }

        // Fetch Status and update progress
        async function fetchStatus() {
            try {
                const response = await fetch('/api/status');
                const status = await response.json();
                
                allResults = status.results;
                updateResultsTable();
                
                const runBtn = document.getElementById('runBtn');
                const stopBtn = document.getElementById('stopBtn');
                const progressContainer = document.getElementById('progressContainer');
                const progressFill = document.getElementById('progressFill');
                const progressLabel = document.getElementById('progressLabel');
                const progressPercent = document.getElementById('progressPercent');
                
                if (status.running) {
                    runBtn.disabled = true;
                    stopBtn.style.display = 'inline-flex';
                    progressContainer.style.display = 'block';
                    
                    const percent = status.total > 0 ? Math.round((status.current / status.total) * 100) : 0;
                    progressFill.style.width = percent + '%';
                    progressLabel.textContent = `Processing claims: ${status.current} / ${status.total}`;
                    progressPercent.textContent = percent + '%';
                    
                    if (!pollingInterval) {
                        pollingInterval = setInterval(fetchStatus, 1500);
                    }
                } else {
                    runBtn.disabled = false;
                    stopBtn.style.display = 'none';
                    
                    if (status.total > 0 && status.current === status.total) {
                        // Completed successfully
                        progressFill.style.width = '100%';
                        progressLabel.textContent = `Run complete! ${status.current} claims verified.`;
                        progressPercent.textContent = '100%';
                    } else if (status.total > 0) {
                        // Stopped/cancelled
                        progressLabel.textContent = `Run stopped at ${status.current} / ${status.total}`;
                    } else {
                        progressContainer.style.display = 'none';
                    }
                    
                    if (pollingInterval) {
                        clearInterval(pollingInterval);
                        pollingInterval = null;
                        // Final refresh models list to show any newly exhausted models
                        fetchModels();
                    }
                }
            } catch (err) {
                console.error("Error checking status:", err);
            }
        }

        // Start Pipeline Trigger
        async function startPipeline() {
            const model = document.getElementById('modelSelect').value;
            try {
                const res = await fetch(`/api/run?model=${model}`, { method: 'POST' });
                const data = await res.json();
                if (data.status === 'success') {
                    // Immediate poll start
                    fetchStatus();
                    if (!pollingInterval) {
                        pollingInterval = setInterval(fetchStatus, 1500);
                    }
                } else {
                    alert(data.message);
                }
            } catch (err) {
                alert("Failed to start pipeline: " + err.message);
            }
        }

        // Cancel Pipeline Run
        async function stopPipeline() {
            try {
                const res = await fetch('/api/stop', { method: 'POST' });
                const data = await res.json();
                if (data.status === 'success') {
                    fetchStatus();
                } else {
                    alert(data.message);
                }
            } catch (err) {
                alert("Failed to stop pipeline: " + err.message);
            }
        }

        // Render table results with filtering
        function updateResultsTable() {
            const body = document.getElementById('resultsBody');
            const filterObj = document.getElementById('filterObject').value;
            const filterStat = document.getElementById('filterStatus').value;
            
            const filtered = allResults.filter(r => {
                if (filterObj && r.claim_object !== filterObj) return false;
                if (filterStat && r.claim_status !== filterStat) return false;
                return true;
            });
            
            if (filtered.length === 0) {
                body.innerHTML = `
                    <tr>
                        <td colspan="7" class="empty-state">
                            <div class="empty-state-icon">🔍</div>
                            <div>No matching evaluation records found.</div>
                        </td>
                    </tr>
                `;
                return;
            }
            
            body.innerHTML = '';
            filtered.forEach(r => {
                const row = document.createElement('tr');
                
                // Image gallery
                let imgsHtml = '<div class="thumbnail-gallery">';
                r.image_paths.forEach((p, idx) => {
                    imgsHtml += `<img src="/images/${p}" class="thumbnail" onclick="showDrawer(${JSON.stringify(r).replace(/"/g, '&quot;')})" title="Click to view full detail">`;
                });
                imgsHtml += '</div>';
                
                // Badges
                const statusBadge = `<span class="badge badge-${r.claim_status}">${r.claim_status.replace(/_/g, ' ')}</span>`;
                const cachedBadge = r.cached ? '<span class="badge badge-cached">Cached</span>' : '';
                const objectBadge = `<span class="badge badge-object">${r.claim_object}</span>`;
                
                row.innerHTML = `
                    <td style="font-weight: 600;">#${r.case_number}</td>
                    <td>${r.user_id}</td>
                    <td>${objectBadge}</td>
                    <td>${statusBadge}${cachedBadge}</td>
                    <td style="text-transform: capitalize;">${r.severity}</td>
                    <td>${imgsHtml}</td>
                    <td>
                        <button class="btn btn-primary" style="padding: 0.35rem 0.75rem; font-size: 0.75rem; width: auto;" onclick='showDrawer(${JSON.stringify(r).replace(/"/g, '&quot;')})'>
                            Details
                        </button>
                    </td>
                `;
                body.appendChild(row);
            });
        }

        function applyFilters() {
            updateResultsTable();
        }

        // Side Drawer Operations
        function showDrawer(row) {
            document.getElementById('drawerCaseTitle').textContent = `Case #${row.case_number} - ${row.user_id}`;
            document.getElementById('drawerChat').textContent = row.user_claim;
            document.getElementById('drawerJustification').textContent = row.claim_status_justification || "No justification returned by model.";
            document.getElementById('drawerPart').textContent = `${row.claim_object.toUpperCase()} / ${row.object_part.replace(/_/g, ' ')}`;
            document.getElementById('drawerSeverity').textContent = row.severity.toUpperCase();
            document.getElementById('drawerValidImage').textContent = row.valid_image.toUpperCase();
            document.getElementById('drawerStandardMet').textContent = row.evidence_standard_met.toUpperCase();
            
            // Render images in drawer
            const imgContainer = document.getElementById('drawerImages');
            imgContainer.innerHTML = '';
            row.image_paths.forEach((p, idx) => {
                const basename = p.split('/').pop().split('.').shift();
                imgContainer.innerHTML += `
                    <div class="drawer-img-container">
                        <img src="/images/${p}" class="drawer-img" onclick="window.open('/images/${p}', '_blank')">
                        <span class="drawer-label-badge">${basename}</span>
                    </div>
                `;
            });
            
            // Risk Flags Badges
            const flagsContainer = document.getElementById('drawerRiskFlags');
            flagsContainer.innerHTML = '';
            if (row.risk_flags && row.risk_flags !== 'none') {
                row.risk_flags.split(';').forEach(f => {
                    const badge = document.createElement('span');
                    badge.className = 'badge badge-contradicted';
                    badge.style.textTransform = 'none';
                    badge.textContent = f.replace(/_/g, ' ');
                    flagsContainer.appendChild(badge);
                });
            } else {
                flagsContainer.innerHTML = '<span style="font-size: 0.875rem; color: var(--text-secondary)">No risks identified.</span>';
            }
            
            document.getElementById('drawer').classList.add('open');
            document.getElementById('overlay').classList.add('open');
        }

        function closeDrawer() {
            document.getElementById('drawer').classList.remove('open');
            document.getElementById('overlay').classList.remove('open');
        }

        // Initialize Page
        window.addEventListener('DOMContentLoaded', () => {
            fetchModels();
            fetchStatus();
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)
