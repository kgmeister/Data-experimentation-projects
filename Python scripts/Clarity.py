# Clarity.py — Staff-facing (no Nibbles)
# - Plain-text replies (no markdown)
# - Persona: 20+ yrs FMCG consultant; ends with a professional-opinion segue and 2–3 short paragraphs
# - Hardcoded Q&A overrides for specific questions (Q1–Q6) with 3s pause
# - Higher token limits to avoid truncation
# - LLM engagement indicator via /health
# - Session resets on page load; no Reset button

from __future__ import annotations
import os, sys, json, re, time
from pathlib import Path
from typing import Dict, List, Any, Optional

# --- Auto-install missing dependencies (no-op if already installed) ---
# Place this block after stdlib imports and BEFORE any 3rd-party imports.
try:
    import importlib, subprocess, sys

    def _ensure_deps():
        # (module_import_name, [pip package(s) to install])
        deps = [
            ("flask",        ["flask"]),
            ("flask_cors",   ["flask-cors"]),
            ("pandas",       ["pandas"]),
            ("numpy",        ["numpy"]),
            ("requests",     ["requests"]),
            ("dotenv",       ["python-dotenv"]),  # your code tries to import `dotenv`
            # For reading Excel files in DataManager:
            ("openpyxl",     ["openpyxl"]),       # .xlsx
            ("xlrd",         ["xlrd"]),           # .xls
        ]
        for mod, pkgs in deps:
            try:
                importlib.import_module(mod)
            except Exception:
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])
                except Exception:
                    # Fallback: bootstrap pip if environment is missing it
                    import ensurepip
                    ensurepip.bootstrap()
                    subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])

    _ensure_deps()
except Exception as _e:
    print(f"[deps] Warning: {_e}")
# --- End auto-install block ---

from flask import Flask, request, jsonify, session
from flask_cors import CORS

import pandas as pd
import numpy as np
import requests

# ----------- Env loading -----------
try:
    from dotenv import dotenv_values
except Exception:
    dotenv_values = None

APP_NAME = "Clarity"
SECRET_KEY = os.environ.get("CLARITY_SECRET_KEY", os.urandom(24))
DEFAULT_ENV_FILE = os.environ.get("ENV_FILE", "bitdeer.env")

DEFAULT_MODEL = os.environ.get("MODEL", "gpt-4o-mini")
DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", "./Team_Cashew_Synthetic_Data")
DEFAULT_API_BASE = os.environ.get("OPENAI_BASE_URL") or os.environ.get("API_URL") or "https://api-inference.bitdeer.ai/v1"
DEFAULT_API_KEY = os.environ.get("OPENAI_API_KEY") or os.environ.get("API_KEY") or ""

# ----------- Data Manager -----------
class DataManager:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.tables: Dict[str, pd.DataFrame] = {}
        self.load_errors: List[str] = []
        self._load_all()

    def _load_all(self):
        if not self.data_dir.exists():
            self.load_errors.append(f"DATA_DIR not found: {self.data_dir}")
            return
        patterns = ["**/*.csv", "**/*.xlsx", "**/*.xls"]
        files: List[Path] = []
        for p in patterns:
            files.extend(self.data_dir.glob(p))
        files = sorted({f.resolve() for f in files})
        for f in files:
            try:
                key = self._key_from_path(f)
                if f.suffix.lower() == ".csv":
                    df = pd.read_csv(f)
                else:
                    df = pd.read_excel(f, sheet_name=0)
                df = self._clean_df(df)
                self.tables[key] = df
            except Exception as e:
                self.load_errors.append(f"Failed to load {f}: {e}")

    def _key_from_path(self, f: Path) -> str:
        base = f.stem.lower()
        base = re.sub(r"[^a-z0-9_]+", "_", base)
        return base

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [str(c).strip() for c in df.columns]
        return df

    def to_schema_prompt(self, max_cols: int = 30) -> str:
        chunks = ["You have access to the following in-memory tables (from CSV/Excel):"]
        for name, df in self.tables.items():
            cols = ", ".join(map(str, df.columns[:max_cols]))
            chunks.append(f"• {name} [{len(df)} rows]: {cols}")
        if self.load_errors:
            chunks.append("Load notes: " + "; ".join(self.load_errors[:5]))
        return "\n".join(chunks)

# ----------- LLM Client -----------
class LLMClient:
    def __init__(self, api_base: str, api_key: str, model: str):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.endpoint = f"{self.api_base}/chat/completions"

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 3000) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(self.endpoint, headers=headers, data=json.dumps(payload), timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            raise RuntimeError(f"Unexpected LLM response: {data}")

# ----------- Persona / Behavior -----------
SYSTEM_PROMPT = (
    "You are a Management Consultant with 20+ years of experience in FMCG. "
    "You specialize in supply chain, sales, digital marketing, and retail operations. "
    "Audience is managers, decision makers, and the C-suite. "
    "Use ONLY the provided in-memory tables and the user's question to compute answers. "
    "Be elaborate and verbose in your insights. "
    "NO MARKDOWN and NO HTML. Do not use bullets, asterisks, code fences, or emojis. "
    "Write plain text sentences with clear line breaks. "
    "At the end of every response, add a professional-opinion section introduced by a natural segue such as "
    "'In my view,' 'I feel that,' or 'What I suggest is that,' followed by two to three short paragraphs "
    "that synthesise implications, trade-offs, and next-step recommendations. "
)

STARTUP_INSIGHTS_PROMPT = (
    "Generate a concise executive summary of key business insights from the available tables for the last 12–24 months "
    "if date fields exist. Keep to a few short paragraphs. No markdown or bullets. "
    "Finish with a professional-opinion section introduced by a segue like 'In my view,' followed by two short paragraphs."
)

# ----------- Flask App -----------
app = Flask(__name__)
app.secret_key = SECRET_KEY
CORS(app, resources={r"/*": {"origins": "*"}})

ENV: Dict[str, str] = {}
DM: Optional[DataManager] = None
LLM: Optional[LLMClient] = None

# LLM engagement indicator
LLM_ENGAGED: bool = False
LLM_LAST_ERROR: Optional[str] = None

def load_env_chain() -> Dict[str, str]:
    merged: Dict[str, str] = {}
    here = Path.cwd()

    def merge_env_file(p: Path):
        nonlocal merged
        if p.exists() and p.is_file():
            try:
                vals = dotenv_values(str(p)) if dotenv_values else {}
                for k, v in (vals or {}).items():
                    if v is None:
                        continue
                    merged.setdefault(k, v)
            except Exception:
                pass

    merge_env_file(here / DEFAULT_ENV_FILE)
    merge_env_file(here / ".env")
    for k, v in os.environ.items():
        merged.setdefault(k, v)

    merged.setdefault("MODEL", DEFAULT_MODEL)
    merged.setdefault("DATA_DIR", DEFAULT_DATA_DIR)
    merged.setdefault("OPENAI_BASE_URL", DEFAULT_API_BASE)
    merged.setdefault("OPENAI_API_KEY", DEFAULT_API_KEY)
    return merged

def probe_llm():
    global LLM_ENGAGED, LLM_LAST_ERROR
    if not LLM:
        LLM_ENGAGED = False
        LLM_LAST_ERROR = "LLM client not initialised"
        return
    try:
        _ = LLM.chat(
            messages=[
                {"role": "system", "content": "Connection probe. Reply with a single character."},
                {"role": "user", "content": "ping"}
            ],
            temperature=0.0,
            max_tokens=1
        )
        LLM_ENGAGED = True
        LLM_LAST_ERROR = None
    except Exception as e:
        LLM_ENGAGED = False
        LLM_LAST_ERROR = str(e)[:300]

def bootstrap_once():
    global ENV, DM, LLM
    if ENV:
        return
    ENV = load_env_chain()
    data_dir = ENV.get("DATA_DIR", DEFAULT_DATA_DIR)
    api_base = ENV.get("OPENAI_BASE_URL") or ENV.get("API_URL") or DEFAULT_API_BASE
    api_key  = ENV.get("OPENAI_API_KEY") or ENV.get("API_KEY") or DEFAULT_API_KEY
    model    = ENV.get("MODEL", DEFAULT_MODEL)
    DM = DataManager(data_dir)
    LLM = LLMClient(api_base=api_base, api_key=api_key, model=model)
    probe_llm()
    print(f"[Clarity] LLM engaged: {LLM_ENGAGED}" + ("" if LLM_ENGAGED else f" | last_error={LLM_LAST_ERROR}"))

@app.before_request
def _ensure_boot():
    bootstrap_once()


# ------------- Helpers -------------
def get_history() -> List[Dict[str, str]]:
    return session.setdefault("history", [])

def set_history(hist: List[Dict[str, str]]):
    session["history"] = hist

def build_system_message() -> Dict[str, str]:
    schema = DM.to_schema_prompt() if DM else "(No data tables)"
    content = f"{SYSTEM_PROMPT}\n\n{schema}"
    return {"role": "system", "content": content}

def startup_insights() -> str:
    messages = [build_system_message(), {"role": "user", "content": STARTUP_INSIGHTS_PROMPT}]
    try:
        # raised max_tokens to reduce truncation
        return LLM.chat(messages, temperature=0.2, max_tokens=1200)
    except Exception as e:
        return f"(Startup insights unavailable: {e})"

# ------------- Routes -------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "app": APP_NAME,
        "status": "ok",
        "model": ENV.get("MODEL"),
        "data_dir": ENV.get("DATA_DIR"),
        "tables": list((DM.tables.keys() if DM else [])),
        "load_errors": (DM.load_errors if DM else []),
        "llm_engaged": LLM_ENGAGED,
        "llm_last_error": LLM_LAST_ERROR,
    })

@app.route("/reset", methods=["POST"])
def reset():
    set_history([])
    return jsonify({"ok": True, "message": "Session cleared."})

@app.route("/chat", methods=["POST"])
def chat():
    global LLM_ENGAGED, LLM_LAST_ERROR
    payload = request.get_json(force=True, silent=True) or {}
    user_msg: str = str(payload.get("message", "")).strip()
    if not user_msg:
        return jsonify({"error": "Empty message"}), 400


    # Normal LLM flow
    hist = get_history()
    messages: List[Dict[str, str]] = [build_system_message()]
    if not hist:
        intro = startup_insights()
        hist.append({"role": "assistant", "content": intro})

    messages.extend(hist)
    messages.append({"role": "user", "content": user_msg})

    try:
        # raised max_tokens to reduce truncation
        reply = LLM.chat(messages, temperature=0.1, max_tokens=2200)
        LLM_ENGAGED = True
        LLM_LAST_ERROR = None
    except Exception as e:
        LLM_ENGAGED = False
        LLM_LAST_ERROR = str(e)[:300]
        err = f"LLM error: {e}"
        return jsonify({"error": err}), 500

    hist.append({"role": "user", "content": user_msg})
    hist.append({"role": "assistant", "content": reply})
    if len(hist) > 30:
        hist[:] = hist[-30:]
    set_history(hist)

    return jsonify({"reply": reply, "history_len": len(hist)})

# ------------- Minimal UI (plain text only; no markdown transforms) -------------
INDEX_HTML = r"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Clarity — Staff Assistant</title>
  <style>
    :root { --bg:#0f172a; --fg:#e2e8f0; --muted:#94a3b8; --card:#111827; --acc:#22d3ee; }
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial; background:var(--bg); color:var(--fg); }
    .wrap { max-width: 900px; margin: 0 auto; padding: 24px; }
    .brand { font-weight: 800; letter-spacing: 0.3px; font-size: 22px; }
    .card { background: var(--card); border: 1px solid #1f2937; border-radius: 16px; padding: 16px; box-shadow: 0 8px 30px rgba(0,0,0,0.25); }
    .log { height: 60vh; overflow:auto; display:flex; flex-direction:column; gap:12px; padding:8px; }
    .msg { padding:12px 14px; border-radius: 14px; line-height: 1.5; }
    .user { background:#0b1324; align-self:flex-end; }
    .bot  { background:#0b1f2a; border:1px solid #1e293b; }
    .sys  { color:var(--muted); font-style: italic; }
    .row { display:flex; gap:8px; margin-top:12px; }
    textarea { flex:1; background:#0b1324; color:var(--fg); border:1px solid #1f2937; border-radius:12px; padding:10px; resize: vertical; min-height: 44px; }
    button { background:var(--acc); color:#001018; border:0; font-weight:700; padding:10px 14px; border-radius: 12px; cursor:pointer; }
    button:disabled { opacity: 0.6; cursor: not-allowed; }
    .small { color:var(--muted); font-size:12px; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="brand">Clarity — Staff Assistant</div>

    <div class="card" style="margin-top:12px;">
      <div id="log" class="log"></div>
      <div class="row">
        <textarea id="box" placeholder="Ask about sales, campaigns, stock…"></textarea>
        <button id="send">Send</button>
      </div>
    </div>
  </div>

  <script>
    const log = document.getElementById('log');
    const box = document.getElementById('box');
    const send = document.getElementById('send');

    function esc(s){
      return s.replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;").replace(/"/g,"&quot;").replace(/'/g,"&#39;");
    }
    function add(role, content){
      const d = document.createElement('div'); d.className = 'msg ' + (role==='user'?'user': (role==='system'?'sys':'bot'));
      d.innerHTML = esc(content).replace(/\n/g, '<br/>');
      log.appendChild(d); log.scrollTop = log.scrollHeight;
    }

    async function sendMsg(){
      const t = box.value.trim(); if(!t) return; box.value='';
      add('user', t); send.disabled=true;
      try{
        const r = await fetch('/chat', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({message:t})});
        const j = await r.json();
        if(j.error){ add('system', j.error); }
        else { add('assistant', j.reply); }
      }catch(e){ add('system', String(e)); }
      send.disabled=false; box.focus();
    }

    // Fresh server session each load
    document.addEventListener('DOMContentLoaded', async ()=>{
      try { await fetch('/reset', {method:'POST'}); } catch(e) {}
      add('system', 'Clarity is ready. Ask a question to begin.');
    });

    send.addEventListener('click', sendMsg);
    box.addEventListener('keydown', (e)=>{ if(e.key==='Enter' && !e.shiftKey){ e.preventDefault(); sendMsg(); }});
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return INDEX_HTML

# ------------- Entrypoint -------------
if __name__ == "__main__":
    bootstrap_once()
    print(f"Loaded tables: {list((DM.tables.keys() if DM else []))}")
    print(f"LLM engaged: {LLM_ENGAGED}" + ("" if LLM_ENGAGED else f" | last_error={LLM_LAST_ERROR}"))
    port = int(os.environ.get("CLARITY_PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=bool(os.environ.get("DEBUG")))
