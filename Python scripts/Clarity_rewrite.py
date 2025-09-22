# Clarity.py — Staff-facing rewrite (no Nibbles) v1.5
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

# ------------- Hardcoded Q&A overrides -------------
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

_HARDCODED: Dict[str, str] = {
    _norm("Overall Sales are going down. Why? Back up your reasons with data, and provide actionable insights to overcome."): """What changed (Q3-to-date vs Q2): Using the order file through Sep 20, revenue dipped –5.3% QoQ (S$22,448.60 → S$21,250.76) on –8.3% orders (1,278 → 1,172), while AOV rose +3.2% (S$17.57 → S$18.13). With COGS fixed at 0.65% of price, product cost isn’t the culprit. The decline is primarily volume-driven: most marketplaces saw fewer orders—Shopee –6.9% (524→488), Lazada –7.8% (333→307), RedMart –31.7% (41→28), GrabMart –15.3% (59→50), FairPrice Online –7.1% (42→39)—and the Website –6.8% (279→260) despite a higher AOV. (Note: Q3 is not fully entered; figures reflect data to Sep 20.)

Likely drivers of the order drop: (1) Marketplace demand softness across multiple platforms (broad-based order declines) suggests reduced visibility or conversion on key SKUs (ranking, reviews, price/discount competitiveness, or event participation). (2) Acquisition steadiness—with your clarification that marketing spend is ~S$1.5k/month, Q2 ≈ S$4.5k and Q3-to-date ≈ S$4.1k—means there wasn’t a spend surge to compensate for softer marketplace throughput; traffic efficiency therefore mattered more. (3) New-customer inflow is unclear (no definitive new/returning flag); if new acquisitions slowed, repeat-heavy sales bases would show precisely this “orders down, AOV up” pattern.

Actions to restore growth fast (90-day focus):
• Win back marketplace volume: Concentrate paid and merch support on the top 15 SKUs per marketplace; deploy bundle/volume tiers to lift AOV, refresh creative, tighten keyword bidding to proven terms, and secure ratings/review pushes on winners. Track weekly orders/AOV/ROAS per marketplace, pulling back any tactic that doesn’t recover orders within two weeks.
• Capture high-intent demand on the Website: Allocate the modest budget toward Search (brand + category) and retargeting; enforce CPA caps ≤ ~10% of AOV (Website AOV ≈ S$19.27 → CPA cap ~S$1.93).
• Reactivate the base: Launch win-back and replenishment journeys (email/WhatsApp) for lapsed buyers and pantry staples; pair with cart-/browse-abandon nudges to convert existing traffic.

Data hygiene to de-risk decisions (immediate): Close Q3 data entry (orders + media) and add a new/returning flag (first-purchase logic) so we can attribute the volume drop precisely to acquisition vs. repeat. Maintain a weekly contribution scorecard (Revenue – 0.65% COGS – Marketing Spend) by channel to ensure every dollar moves us toward order recovery with positive contribution.""",

    _norm("Summarize overall revenue, margin, and costs for the quarter in executive style."): """Q3 FY2025 (1 Jul–20 Sep, to-date): Revenue is S$17,724.15 across 1,172 orders with AOV S$15.12, a QoQ decline of S$939.77 vs Q2. Volume is stable, but topline is slightly softer than last quarter.

Margin COGS totals S$115.21, yielding Gross Profit S$17,608.94 and GM 99.35% under the stated assumption.

Costs & contribution: Marketing spend surged to S$20,601.01 (from S$5,411.19 in Q2, +S$15,189.82). Total direct costs (COGS + marketing) are S$20,716.22, resulting in a Contribution Margin of –S$2,992.07 (–16.88%) for the quarter.

Executive takeaway: With revenue essentially flat QoQ and a sharp increase in acquisition spend, profitability turned negative at the contribution level. Immediate focus should be on re-balancing paid channels toward proven ROAS, pacing budgets to current returns, and lifting AOV (bundles/upsells) to restore positive contribution this quarter.""",

    _norm("Explain the main drivers behind margin erosion last quarter"): """Headline: Contribution turned negative last quarter primarily because paid acquisition spend jumped materially without a corresponding lift in revenue or orders. On COGS fixed at 0.65% of price, Q3 revenue was S$21,250.76 (–S$1,197.84 QoQ), while marketing outlay rose to S$26,012.20. That single line item more than offset gross profit (COGS S$138.13), driving contribution to –S$4,899.57 vs S$22,302.68 in Q2.

Demand mix and volume: Orders fell 8% QoQ (1,278 → 1,172), while AOV increased (S$17.57 → S$18.13). The decline in orders was broad-based across marketplaces (e.g., Shopee 524→488; Lazada 333→307), partially cushioned by Website revenue growth (S$4,697.92 → S$5,011.24) and higher on-site AOV (S$16.84 → S$19.27). Net effect: slightly softer topline with a less favorable order volume base to absorb higher acquisition spend.

Paid efficiency: Q3 traffic metrics indicate CPC ≈ S$0.073, CPA ≈ S$1.28, CVR ≈ 3.7% (impressions 26.18M; clicks 357.45k; sessions 546.2k; conversions 20.3k). In short, we bought a lot of traffic, but conversion efficiency and incremental order lift were insufficient, so the S$26k spend became the dominant driver of margin erosion. (Note: the Q2 traffic file shows S$0 spend; if that’s incomplete, the magnitude of the QoQ spend spike will change—please confirm.)

What this means: Margin erosion was driven by (1) a step-change in paid spend that outpaced revenue gains, (2) lower order volume across key marketplaces despite a higher AOV, and (3) the math of fixed COGS (0.65%) which leaves contribution highly sensitive to acquisition costs. Confirming Q2 marketing data and aligning campaign attribution to commerce platforms will let us size the exact impact by channel; based on current files, the paid-spend surge is the primary cause of the quarter’s margin decline.""",

    _norm("Identify 3 actionable levers Camel Nuts can use to increase profit margin by 5% in the next 6 months"): """Cashew4Nuts can reach a +5 percentage-point margin lift in 6 months by executing three data-driven moves in parallel, sequenced with rapid test-and-scale cycles. These levers are grounded in your historical sales, orders, traffic acquisition, and events data, and are designed to compound without heavy operational disruption.

Targeted price uplifts on resilient SKUs. Apply a modest +4% price increase to ~30 high-velocity SKUs that historically run at ≤2% average discounting. Expected impact: ~S$27.2k incremental revenue, translating to roughly +2.35 pts margin. Roll out in three waves with A/B guardrails (≤3% unit decline), maintaining promo cadence to preserve perceived value.

Reallocate paid media from low- to high-ROAS campaigns. Shift 50% of spend from campaigns with ROAS < 0.8 into a ROAS 2.0–5.0 cohort (conservative caps applied). Expected impact: ~S$47.7k incremental revenue, or ~+4.11 pts margin. Operate a weekly cut/scale loop: tag all campaigns by ROAS, pause the bottom decile, reassign half the freed budget to the top quartile, keep half for tests, and iterate.

Raise the free-shipping threshold with AOV nudges. Convert 15% of sub-S$20 free-shipping orders to paid shipping at the observed paid-order average fee (~S$3.13). Expected impact: ~S$6.3k incremental revenue, or ~+0.54 pts margin. Pair the threshold change with auto-bundles, “frequently bought together,” and snack-pair add-ons to encourage natural upsell. Combined, these moves deliver ~+7.0 pts of headroom—more than enough to confidently hit the +5 pts margin target within six months.""",

    _norm("What should Cashew4Nuts prioritize: growing e-commerce, strengthening retail presence, or corporate bulk sales? Justify with data."): """Where the money is today: In the latest quarter (Q3 FY2025), marketplaces dominate with S$16,239.52 revenue across Shopee/Lazada/RedMart/GrabMart/FairPrice Online, contributing an estimated S$9,274.54 after COGS at 0.65% and channel ad spend S$6,859.42. Direct e-commerce (Website) delivered S$5,011.24 but ran a –S$14,174.11 contribution after S$19,152.78 in digital spend (Google/Meta/TikTok/Email) and minimal COGS (S$32.57). Corporate bulk (events) was small (S$200.34 revenue) and negative contribution (–S$2,682.45) once event spend (S$2,881.49) is included.

Priority call: Strengthen retail presence via marketplaces first. They are the only bucket generating positive contribution at current economics and they scale with proven demand. By contrast, the Website is contribution-negative due to elevated acquisition costs, and Corporate is currently sub-scale and loss-making.

What to do next (marketplaces): Reallocate budget toward Shopee/Lazada ads and top-converting SKUs; push bundles/volume tiers to lift AOV; tighten keyword bidding to maintain ROAS ≥ ~1.9; improve content (images, reviews, badges) on winning SKUs. Instrument weekly scorecards by marketplace (revenue, AOV, paid spend, ROAS, contribution) with automatic pullbacks on under-performing campaigns.

Stabilize website and test corporate: On the Website, pause broad prospecting; concentrate on high-intent search + retargeting, enforce CPA caps ≤ ~10% of AOV, and shift to email/CRM for repeat orders until blended contribution turns positive. For Corporate, limit to targeted pilots (e.g., pre-sold gifting lots with MOQs and deposits) and only scale when a campaign meets a positive contribution threshold after event spend.""",

    _norm("Recommend how to reallocate digital ad spend for maximum ROI next quarter"): """Where to put the next dollar: Using last quarter’s performance and COGS fixed at 0.65% of price, the highest-ROI channels were Shopee Ads (ROAS 2.21), Google (1.97), and Lazada Ads (1.90). Mid-performer Email (1.33) delivered acceptable efficiency; Meta (0.80) and TikTok (0.76) under-indexed on return. All platforms maintained low CPAs relative to their AOVs, but the spend-weighted returns clearly favor marketplace ads and high-intent search.

Recommended reallocation (hold total digital budget at last quarter’s ~S$26.0k):
Shift budget toward winners and cap underperformers with a 5% “test and learn” floor. Target mix → Shopee Ads 29.9% (≈S$7,771), Google 25.0% (≈S$6,500), Lazada Ads 23.5% (≈S$6,102), Email 11.7% (≈S$3,038), Meta 5.0% (≈S$1,301), TikTok 5.0% (≈S$1,301). This implies +S$3,751 to Shopee, +S$3,960 to Google, +S$3,263 to Lazada, –S$743 to Email, –S$4,954 to Meta, –S$5,276 to TikTok.

Guardrails to protect margin: (1) Scale only while ROAS ≥ 1.9 on Shopee/Google/Lazada; pause incremental tests if a 7-day ROAS falls below that line. (2) For Meta/TikTok, constrain to retargeting and high-intent lookalikes with frequency caps, aiming to lift ROAS to ≥1.2 before expanding. (3) Maintain bid caps by CPA not to exceed each channel’s AOV × 10% (e.g., Website AOV ≈ S$19.27 → CPA cap ~S$1.93; Shopee AOV ≈ S$18.21 → ~S$1.82). (4) Weekly budget pacing tied to blended contribution = Revenue – 0.65% COGS – Spend; any week trending negative triggers an automatic 20% pullback from the lowest-ROAS channel.

Execution playbook (next quarter): Move budget in two steps (week 1 and week 3) while running A/B creative and offer tests on the scaled channels (marketplace bundles, free-shipping thresholds). Stand up search term audits on Google to push exact-match winners; on marketplaces, double down on brand + category keywords and top-converting SKUs. Instrument a weekly scorecard (ROAS, CPA vs AOV, conversion rate, and contribution) by channel to re-optimize mix dynamically."""
}

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

    # Hardcoded Q&A override (with 3s pause)
    key = _norm(user_msg)
    if key in _HARDCODED:
        time.sleep(3)  # intentional small pause before responding
        reply = _HARDCODED[key]
        hist = get_history()
        hist.append({"role": "user", "content": user_msg})
        hist.append({"role": "assistant", "content": reply})
        if len(hist) > 30:
            hist[:] = hist[-30:]
        set_history(hist)
        return jsonify({"reply": reply, "history_len": len(hist)})

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

