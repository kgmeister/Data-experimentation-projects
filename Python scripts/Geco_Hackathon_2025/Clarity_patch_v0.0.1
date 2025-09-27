# Clarity.py — Staff-facing (no Nibbles)
# Patched for API failover
# - Plain-text replies (no markdown)
# - Persona: 20+ yrs FMCG consultant; ends with a professional-opinion segue and 2–3 short paragraphs
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

    def _ensure(pkgs: List[str]):
        for p in pkgs:
            try:
                importlib.import_module(p)
            except Exception:
                subprocess.check_call([sys.executable, "-m", "pip", "install", p])

    # Core web + data + env
    _ensure(["flask", "flask_cors", "pandas", "numpy", "python-dotenv", "requests"])
except Exception as _e:
    try:
        # Minimal fallback installer if importlib failed early
        import subprocess
        pkgs = ["flask", "flask_cors", "pandas", "numpy", "python-dotenv", "requests"]
        subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])
    except Exception as _e2:
        print(f"[deps] Warning: initial auto-install failed: {_e2}")
    else:
        try:
            import importlib
            for p in ["flask", "flask_cors", "pandas", "numpy", "python-dotenv", "requests"]:
                importlib.import_module(p)
        except Exception as _e3:
            print(f"[deps] Warning: {_e3}")
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
DEFAULT_ENV_FILE = os.environ.get("ENV_FILE", "llm.env")

DEFAULT_MODEL = os.environ.get("MODEL", "gpt-4o-mini")
DEFAULT_DATA_DIR = os.environ.get("DATA_DIR", "./Team_Cashew_Synthetic_Data")
DEFAULT_API_BASE = os.environ.get("OPENAI_BASE_URL") or os.environ.get("API_URL") or "https://api.openai.com/v1"
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
            self.load_errors.append(f"Data dir not found: {self.data_dir}")
            return
        exts = {".csv", ".xlsx", ".xls"}
        for f in sorted(self.data_dir.rglob("*")):
            if f.suffix.lower() not in exts:
                continue
            try:
                if f.suffix.lower() == ".csv":
                    df = pd.read_csv(f)
                else:
                    df = pd.read_excel(f)
                key = self._key_from_path(f)
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

    def to_schema_prompt(self, 
                         max_cols: int = 30) -> str:
        chunks = ["You have access to the following in-memory tables (from CSV/Excel):"]
        for name, df in self.tables.items():
            cols = ", ".join(map(str, df.columns[:max_cols]))
            chunks.append(f"• {name} [{len(df)} rows]: {cols}")
        if self.load_errors:
            chunks.append("Load notes: " + "; ".join(self.load_errors[:5]))
        return "\n".join(chunks)

# ----------- LLM Failover (replaces single-provider client) -----------
# Adopted from user's llm_failover.py (inline to keep single-file deployment)
import re as _re
from typing import Tuple

_NUM_URL_KEY = _re.compile(r"^LLM_(\d+)_API_URL$", _re.I)

def _parse_status_list(spec: str):
    out = []
    for part in (spec or "").split(","):
        part = part.strip()
        if not part: continue
        if "-" in part:
            a, b = part.split("-", 1); out.append((int(a), int(b)))
        else:
            v = int(part); out.append((v, v))
    return out

def _status_matches(ranges, status: int) -> bool:
    return any(lo <= status <= hi for lo, hi in ranges)

def _parse_extra_headers(raw: Optional[str]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if not raw: return headers
    for p in [p.strip() for p in raw.split(";") if p.strip()]:
        if ":" in p:
            k, v = p.split(":", 1)
            headers[k.strip()] = v.strip()
    return headers

def _auth_headers(auth_type: str, key: str) -> Dict[str, str]:
    auth_type = (auth_type or "bearer").strip().lower()
    if auth_type == "bearer":
        return {"Authorization": f"Bearer {key}"}
    if auth_type == "api-key":
        return {"api-key": key}
    if auth_type.startswith("header:"):
        name = auth_type.split(":", 1)[1].strip() or "Authorization"
        return {name: key}
    return {"Authorization": f"Bearer {key}"}

def _discover_indices(env: Dict[str, str]) -> List[int]:
    nums = {int(m.group(1)) for k in env.keys() for m in [_NUM_URL_KEY.match(k)] if m}
    if (env.get("API_URL") or env.get("LLM_API_URL")) and (env.get("API_KEY") or env.get("LLM_API_KEY")):
        nums.add(0)
    return sorted(nums)

def _provider_from_env(env: Dict[str, str], n: int) -> Optional[Dict[str, str]]:
    if n == 0:
        api_url = env.get("LLM_API_URL") or env.get("API_URL")
        api_key = env.get("LLM_API_KEY") or env.get("API_KEY")
        model   = (env.get("LLM_MODEL") or env.get("MODEL") or "").strip()
        auth    = (env.get("LLM_AUTH_TYPE") or "bearer").strip().lower()
        extra   = env.get("LLM_HEADERS") or ""
    else:
        p = f"LLM_{n}_"
        api_url = env.get(p + "API_URL")
        api_key = env.get(p + "API_KEY")
        model   = (env.get(p + "MODEL") or "").strip()
        auth    = (env.get(p + "AUTH_TYPE") or "bearer").strip().lower()
        extra   = env.get(p + "HEADERS") or ""
    if not (api_url and api_key):
        return None
    return {"idx": n, "name": f"LLM_{n}", "api_url": api_url.strip(), "api_key": api_key.strip(),
            "model": model, "auth_type": auth, "extra_headers": extra}

def iter_llm_providers() -> List[Dict[str, str]]:
    env = os.environ
    out: List[Dict[str, str]] = []
    for n in _discover_indices(env):
        p = _provider_from_env(env, n)
        if p: out.append(p)
    return out

def _build_request(provider: Dict[str, str], messages, temperature: float, max_tokens: Optional[int]):
    url = provider["api_url"]
    headers = {"Content-Type": "application/json"}
    headers.update(_auth_headers(provider["auth_type"], provider["api_key"]))
    headers.update(_parse_extra_headers(provider["extra_headers"]))
    payload = {"messages": messages, "temperature": temperature}
    if provider["model"]: payload["model"] = provider["model"]
    if max_tokens is not None: payload["max_tokens"] = max_tokens
    return url, headers, payload

def chat_with_failover(messages, temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
    timeout  = float(os.getenv("LLM_TIMEOUT_SECONDS", "25"))
    retries  = int(os.getenv("LLM_RETRIES", "2"))
    backoff  = float(os.getenv("LLM_BACKOFF_SECONDS", "1.2"))
    failover_ranges = _parse_status_list(os.getenv("LLM_FAILOVER_ON_STATUS", "500-599"))
    debug = os.getenv("LLM_FAILOVER_DEBUG", "0") == "1"
    if temperature is None:
        temperature = float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.2"))
    if max_tokens is None:
        mt = os.getenv("LLM_DEFAULT_MAX_TOKENS")
        max_tokens = int(mt) if mt else None
    last_err: Optional[Exception] = None
    for p in iter_llm_providers():
        if debug: print(f"[llm_failover] Trying {p['name']} at {p['api_url']}")
        for attempt in range(retries + 1):
            try:
                url, headers, payload = _build_request(p, messages, temperature, max_tokens)
                r = requests.post(url, json=payload, headers=headers, timeout=timeout)
                if r.status_code == 200:
                    data = r.json()
                    choices = data.get("choices") or []
                    if choices and "message" in choices[0] and "content" in choices[0]["message"]:
                        if debug: print(f"[llm_failover] {p['name']} succeeded (message.content).")
                        return choices[0]["message"]["content"]
                    if choices and "text" in choices[0]:
                        if debug: print(f"[llm_failover] {p['name']} succeeded (text).")
                        return choices[0]["text"]
                    raise RuntimeError(f"{p['name']} unexpected response: {str(data)[:400]}")
                if _status_matches(failover_ranges, r.status_code):
                    last_err = RuntimeError(f"{p['name']} HTTP {r.status_code}: {r.text[:400]}")
                    if debug: print(f"[llm_failover] Failover due to status {r.status_code} from {p['name']}")
                    break
                else:
                    last_err = RuntimeError(f"{p['name']} HTTP {r.status_code}: {r.text[:400]}")
                    if debug: print(f"[llm_failover] Retry {p['name']} (attempt {attempt+1}/{retries}) after status {r.status_code}")
                    time.sleep(backoff * (attempt + 1))
            except requests.RequestException as e:
                last_err = e
                if debug: print(f"[llm_failover] Network error on {p['name']}: {e}. Retry {attempt+1}/{retries}")
                time.sleep(backoff * (attempt + 1))
        if debug: print(f"[llm_failover] Moving past {p['name']} to next provider...")
    if debug: print("[llm_failover] All providers failed.")
    raise last_err or RuntimeError("All LLM providers failed")

# ----------- Prompting -----------
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
    "Generate a concise executive summary of 2–3 short paragraphs using the loaded tables. "
    "Surface notable year-over-year trends, anomalies in channel or SKU performance, and one operational risk. "
    "Avoid bullet points and markdown. Plain text only."
)

# ----------- Flask app setup -----------
app = Flask(APP_NAME)
app.secret_key = SECRET_KEY
CORS(app, supports_credentials=True)

ENV: Optional[Dict[str, Any]] = None
DM: Optional[DataManager] = None
LLM_ENGAGED: bool = False
LLM_LAST_ERROR: Optional[str] = None

def load_env_chain() -> Dict[str, str]:
    """
    Merge env from:
      1) ENV_FILE (default llm.env) if present
      2) .env (optional)
      3) process env
    """
    merged: Dict[str, str] = {}
    here = Path(__file__).resolve().parent

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
    if not DM:
        LLM_ENGAGED = False
        LLM_LAST_ERROR = "Data manager not initialised"
        return
    try:
        _ = chat_with_failover(
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
        LLM_LAST_ERROR = str(e)

def build_system_message() -> Dict[str, str]:
    schema = DM.to_schema_prompt() if DM else "(no tables loaded)"
    content = SYSTEM_PROMPT + "\n\n" + schema
    return {"role": "system", "content": content}

def get_history() -> List[Dict[str, str]]:
    return session.get("history", [])

def set_history(hist: List[Dict[str, str]]):
    session["history"] = hist

def startup_insights() -> str:
    messages = [build_system_message(),
                {"role": "user", "content": STARTUP_INSIGHTS_PROMPT}]
    try:
        # raised max_tokens to reduce truncation
        return chat_with_failover(messages, temperature=0.2, max_tokens=1200)
    except Exception as e:
        return f"(Startup insights unavailable: {e})"

# ------------- Routes -------------
@app.before_request
def _ensure_boot():
    if not session.get("_bootstrapped", False):
        session["_bootstrapped"] = True
        # Force-reset session history on first request of each browser session
        session["history"] = []

@app.route("/health", methods=["GET"])
def health():
    probe_llm()
    return jsonify({
        "ok": LLM_ENGAGED,
        "last_error": LLM_LAST_ERROR,
        "tables": sorted(list((DM.tables.keys() if DM else []))),
        "env_file": DEFAULT_ENV_FILE
    })

@app.route("/debug/sources", methods=["GET"])
def debug_sources():
    dd = {
        "data_dir": (str(DM.data_dir) if DM else None),
        "tables": sorted(DM.tables.keys()) if DM else [],
        "load_errors": DM.load_errors if DM else []
    }
    return jsonify(dd)

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
        reply = chat_with_failover(messages, temperature=0.2, max_tokens=3000)
    except Exception as e:
        LLM_ENGAGED = False
        LLM_LAST_ERROR = str(e)
        return jsonify({"error": f"LLM failure: {e}"}), 502

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
    body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; background:var(--bg); color:var(--fg); }
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
      <div id="log" class="log">
        <div class="msg sys">Session started. Type a question about sales, marketing, or operations.</div>
      </div>
      <div class="row">
        <textarea id="msg" placeholder="Ask about sales trends, SKU mix shifts, marketing ROI, etc."></textarea>
        <button id="send">Send</button>
      </div>
      <div class="small" id="status"></div>
    </div>
  </div>
<script>
const log = document.getElementById('log');
const msg = document.getElementById('msg');
const send = document.getElementById('send');
const statusEl = document.getElementById('status');

function add(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + (role==='user'?'user':'bot');
  div.textContent = text;
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

async function ping() {
  try {
    const r = await fetch('/health');
    const j = await r.json();
    statusEl.textContent = j.ok ? 'LLM: OK' : ('LLM: '+ (j.last_error||'error'));
  } catch (e) {
    statusEl.textContent = 'LLM: error';
  }
}
ping();

send.onclick = async () => {
  const text = msg.value.trim();
  if (!text) return;
  add('user', text);
  msg.value = '';
  send.disabled = true;
  try {
    const r = await fetch('/chat', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({message: text})
    });
    const j = await r.json();
    if (j.reply) add('assistant', j.reply);
    else add('assistant', j.error || 'Error');
  } catch (e) {
    add('assistant', 'Network error');
  } finally {
    send.disabled = false;
  }
};
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return INDEX_HTML

def bootstrap_once():
    global ENV, DM
    if ENV:
        return
    ENV = load_env_chain()
    # Make ENV variables visible to failover block
    try:
        os.environ.update({k:str(v) for k,v in ENV.items() if isinstance(k,str)})
    except Exception:
        pass
    data_dir = ENV.get("DATA_DIR", DEFAULT_DATA_DIR)
    DM = DataManager(data_dir)
    probe_llm()
    print(f"[Clarity] LLM engaged: {LLM_ENGAGED}" + ("" if LLM_ENGAGED else f" | last_error={LLM_LAST_ERROR}"))

@app.before_request
def _ensure_boot():
    bootstrap_once()

# ------------- Entrypoint -------------
if __name__ == "__main__":
    bootstrap_once()
    print(f"Loaded tables: {list((DM.tables.keys() if DM else []))}")
    print(f"LLM engaged: {LLM_ENGAGED}" + ("" if LLM_ENGAGED else f" | last_error={LLM_LAST_ERROR}"))
    port = int(os.environ.get("CLARITY_PORT", 5000))
    app.run(host="127.0.0.1", port=port, debug=bool(os.environ.get("DEBUG")))
