# BrainAuto 🚀

A Python automation framework for the **WorldQuant Brain API** that runs
alpha simulations through a waiting queue with exponential back-off.

---

## Project Structure

```
BrainAuto/
├── main.py               # Entry point
├── config.py             # Credential & URL config (reads from .env)
├── requirements.txt
├── alphas.json           # Sample alpha expressions
├── .env.example          # Copy this to .env and fill in your credentials
│
├── src/
│   ├── client.py         # Brain REST API wrapper
│   ├── automation.py     # Queue + exponential back-off engine
│   ├── utils.py          # Payload builder, JSON/CSV loaders
│   ├── logger.py         # Centralised logger (stdout + file)
│   └── result_logger.py  # Persists results to ./results/ (JSONL + CSV)
│
├── logs/                 # Created at runtime
└── results/              # Created at runtime
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure credentials

```bash
cp .env.example .env
# Open .env and fill in your Brain username and password
```

### 3. Add your alphas

Either edit the `load_alphas()` function in `main.py`, or populate
`alphas.json` and switch to file-based loading:

```python
from src.utils import load_from_json, build_simulation_payload

alphas = [
    {**build_simulation_payload(a["expression"], **{k: v for k, v in a.items() if k not in ["expression","name"]}), "name": a["name"]}
    for a in load_from_json("alphas.json")
]
```

---

## Run

```bash
python main.py
```

---

## How It Works

```
┌─────────────┐     add_tasks()     ┌───────────────────────┐
│  alphas.json │ ─────────────────► │   queue.Queue (FIFO)  │
└─────────────┘                     └───────────┬───────────┘
                                                │  dequeue one
                                                ▼
                                    ┌───────────────────────┐
                                    │  simulate_alpha()     │  ◄─── retry on
                                    │  (POST /simulations)  │       429 / 5xx
                                    └───────────┬───────────┘
                                                │  sim_id
                                                ▼
                                    ┌───────────────────────┐
                                    │  poll status every 5s │
                                    │  until SUCCESS|ERROR  │
                                    └───────────┬───────────┘
                                                │
                                                ▼
                                    ┌───────────────────────┐
                                    │  ResultLogger.record()│
                                    │  → results/run_*.csv  │
                                    │  → results/run_*.jsonl│
                                    └───────────────────────┘
```

### Exponential Back-off

| Attempt | Delay  |
|---------|--------|
| 1       | 2 s    |
| 2       | 4 s    |
| 3       | 8 s    |
| 4       | 16 s   |
| 5       | 32 s   |

Retries are triggered by: `429 Too Many Requests`, `500/502/503/504` server
errors, and network-level exceptions (timeouts, connection resets).

---

## Output

- **Console + `logs/brainsauto.log`** – live progress
- **`results/run_<timestamp>.jsonl`** – one JSON object per alpha
- **`results/run_<timestamp>.csv`** – spreadsheet-friendly summary

---

## Note on the Official SDK

WorldQuant Brain does **not** publish an official Python SDK.
This project uses the REST API documented inside the Brain platform
(under your account's API Docs section).
