# Multi-Modal Evidence Review System

This directory contains the Python codebase for verifying visual damage claims using Google Gemini's multi-modal models, structured claiming rules, user histories, and evidence standards.

## Project Structure

```text
code/
├── main.py                # Main entry point to generate predictions on claims.csv
├── utils.py               # Utilities for loading support CSVs, VLM API calls, caching, and rate limiting
├── prompt_templates.py    # System instructions, strict rules, allowed values, and user prompts
├── README.md              # This documentation file
└── evaluation/
    ├── main.py            # Evaluation pipeline comparing Strategy A vs Strategy B
    └── evaluation_report.md  # Generated evaluation report comparing zero-shot and few-shot strategies
```

## Solution Design

### 1. Robust Model Failover Pool
To protect against strict daily quotas (20 requests per day) on free tier Gemini API projects, the system implements a **failover model pool**. When calling the VLM, if a model hits a `ResourceExhausted` (429) daily limit, it is automatically blacklisted in memory, and the client fails over to the next candidate model:
- `gemini-2.5-flash-lite`
- `gemini-3-flash-preview`
- `gemini-3.1-flash-lite`
- `gemini-flash-lite-latest`
- `gemini-2.5-flash`

### 2. Multi-Tier Rate Limiting
To strictly satisfy the 5 requests-per-minute (RPM) limit of the Gemini free tier:
- Preemptive throttle spaces VLM calls by at least 13 seconds.
- Caught 429 rate limit exceptions trigger an automatic 30-second exponential retry delay.

### 3. Image Hashing Cache
To prevent redundant API usage and eliminate operational costs:
- API inputs (system prompts, user claims, and base64-encoded images) are hashed using SHA-256.
- Cached results are saved locally in `.openai_cache.json`. Repeated executions achieve a **100% cache hit rate** with sub-millisecond response times.

### 4. Input Pre-processing and Verification
- **AVIF Decoders**: Configured Pillow with the `pillow-avif-plugin` to correctly identify and resize AVIF images disguised as `.jpg` in the test set.
- **Risk Flag Propagation**: Injected user history risks (`user_history_risk` and `manual_review_required`) from `user_history.csv` into the VLM output flags.
- **Allowed Values Assertion**: Predictions are programmatically validated against strictly defined option sets (`supported`, `contradicted`, `not_enough_information` for status; frontend parts constraints, etc.).

## Setup & Run

### 1. Prerequisites
Install dependencies:
```bash
pip install google-generativeai python-dotenv pillow pillow-avif-plugin pandas
```

### 2. Configure Environment
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Generate Predictions
To run the verification pipeline on `dataset/claims.csv` and write outputs to `output.csv` (root) and `dataset/output.csv`:
```bash
python code/main.py
```

### 4. Run Evaluation
To evaluate and compare zero-shot vs few-shot strategies on `dataset/sample_claims.csv`:
```bash
python code/evaluation/main.py
```
The metrics comparison will be written to `code/evaluation/evaluation_report.md`.
