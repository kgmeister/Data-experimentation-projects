# Nibbles.py — Version 0.3.9
# - Fix: robust LLM endpoint builder (no more .../chat/completions/chat/completions)
# - Reads SYSTEM_PROMPT from env (falls back to default)
# - /healthz now shows resolved chat_url
# - /reset still clears per-session chat history only
#
# Run:  python Nibbles.py  → http://127.0.0.1:5001
#
# bitdeer.env / .env:
#   DATA_DIR=C:/Team_Cashew_Synthetic_Data
#   COMPANY_NAME=Cashew4Nuts
#   CURRENCY=SGD
#   NIBBLES_PORT=5001
#   LLM_API_URL=https://api-inference.bitdeer.ai/v1       # any of: host, /v1, or full /v1/chat/completions
#   LLM_API_KEY=sk-...
#   LLM_MODEL=gpt-4o-mini
#   SYSTEM_PROMPT=You are Nibbles...
#   LLM_TIMEOUT_SEC=25

from __future__ import annotations
import os, re, sys, json
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from collections import defaultdict

# ---------- bootstrap installs ----------
def _pip_install(pkgs: List[str]):
    import subprocess
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", *pkgs])
    except Exception as e:
        print(f"[warn] Could not auto-install {pkgs}: {e}", file=sys.stderr)

try:
    import pandas as pd
except Exception:
    _pip_install(["pandas>=2.1.0"]); import pandas as pd

try:
    from flask import Flask, request, jsonify, render_template_string
except Exception:
    _pip_install(["flask>=3.0.0"]); from flask import Flask, request, jsonify, render_template_string

try:
    from dotenv import load_dotenv
except Exception:
    _pip_install(["python-dotenv>=1.0.0"]); from dotenv import load_dotenv

try:
    from rapidfuzz import process, fuzz
except Exception:
    _pip_install(["rapidfuzz>=3.5.2"]); from rapidfuzz import process, fuzz

try:
    from unidecode import unidecode
except Exception:
    _pip_install(["Unidecode>=1.3.6"]); from unidecode import unidecode

try:
    import requests
except Exception:
    _pip_install(["requests>=2.31.0"]); import requests

# ---------- env loading ----------
def load_env():
    custom = os.environ.get("ENV_FILE")
    if custom and os.path.exists(custom):
        load_dotenv(custom, override=True); return
    here = Path(__file__).resolve().parent
    for fn in ("bitdeer.env", ".env"):
        p = here / fn
        if p.exists(): load_dotenv(p, override=True); return
load_env()

CURRENCY     = os.environ.get("CURRENCY", "SGD").strip()
COMPANY_NAME = os.environ.get("COMPANY_NAME", "Cashew4Nuts").strip()
DATA_DIR     = os.environ.get("DATA_DIR", str(Path(__file__).resolve().parent))
PORT         = int(os.environ.get("NIBBLES_PORT", "5001"))

LLM_API_URL  = (os.environ.get("LLM_API_URL") or os.environ.get("OPENAI_API_BASE") or "").strip()
LLM_API_KEY  = (os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
LLM_MODEL    = (os.environ.get("LLM_MODEL")   or os.environ.get("OPENAI_MODEL")   or "").strip()
LLM_TIMEOUT  = int(os.environ.get("LLM_TIMEOUT_SEC", "25"))

def build_chat_url(base: str) -> str:
    """
    Accepts:
      - https://host
      - https://host/v1
      - https://host/openai/v1
      - https://host/v1/chat/completions (already complete)
      - https://host/chat/completions (already complete)
      - https://host/v1/responses (alternative)
    Returns a valid /chat/completions endpoint. Defaults to OpenAI if base missing.
    """
    if not base:
        return "https://api.openai.com/v1/chat/completions"
    b = base.rstrip("/")
    # Already a full endpoint?
    if re.search(r"/(chat/)?completions$", b) or b.endswith("/responses"):
        return b
    # If base already includes a version segment (/v1, /v2, /openai/v1, etc.), append chat/completions
    if re.search(r"/v\d+($|/)", b):
        return b + "/chat/completions"
    # Otherwise assume host-only
    return b + "/v1/chat/completions"

CHAT_URL     = build_chat_url(LLM_API_URL)
LLM_ENABLED  = bool(CHAT_URL and LLM_API_KEY and LLM_MODEL)

# ---------- helpers ----------
def find_file(filename_like: str, base_dir: str) -> Optional[Path]:
    base = Path(base_dir)
    if base.is_file():
        return base if base.name.lower() == filename_like.lower() else None
    for p in base.rglob("*"):
        if p.is_file() and p.name.lower() == filename_like.lower():
            return p
    return None

PRICE_COLS = ["unit_price_sgd","unit_price","price_sgd","price","list_price_sgd","list_price","net_price_sgd"]
DESC_COLS  = ["sku_description","product_name","name","title","description"]
PACK_COLS  = ["pack_size","packsize","size","weight_g","weight","net_weight_g","net_weight"]
SKU_COLS   = ["sku","item_code","product_code","item_no","item","sku_id"]

def first_existing(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns: return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c in lower_map: return lower_map[c]
    return None

_SIZE_PATTERNS = [
    r"(?P<num>\d+(?:[.,]\d+)?)\s*(?P<u>kg|kilogram|kilo|公斤|千克|กก|キロ|킬로|кг|كيلو|กิโλο)",
    r"(?P<num>\d+(?:[.,]\d+)?)\s*(?P<u>g|gram|克|公克|グラム|그램|กรัม)",
    r"(?P<num>\d+(?:[.,]\d+)?)(?P<u>kg|g)",
]
_SIZE_RE = [re.compile(p, re.IGNORECASE) for p in _SIZE_PATTERNS]

def parse_grams(text: str) -> Optional[int]:
    t = str(text)
    for rx in _SIZE_RE:
        m = rx.search(t)
        if not m: continue
        num = m.group("num").replace(",", ".")
        unit = m.group("u").lower()
        try:
            val = float(num)
        except:
            continue
        if unit.startswith("kg") or unit in {"kilogram","kilo","公斤","千克","กก","キロ","킬로","кг","كيلو","กิโλο"}:
            return int(round(val * 1000))
        return int(round(val))
    return None

def normalize_text(s: str) -> str:
    s = unidecode(str(s)).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

# ---------- multilingual tokens ----------
SYNONYMS: Dict[str, List[str]] = {
    # families
    "peanut":   ["花生","落花生","ピーナッツ","ﾋﾟｰﾅｯﾂ","땅콩","kacang","mani","peanuts","peanut"],
    "almond":   ["杏仁","扁桃","アーモンド","아몬드","almendra","badam","almonds","almond"],
    "cashew":   ["腰果","カシューナッツ","캐슈넛","kacang mete","kasuy","cashews","cashew"],
    "pistachio":["开心果","開心果","ピスタチオ","피스타치오","pista","pistachios","pistachio"],
    "macadamia":["夏威夷果","マカダミア","마카다미아","macadamias","macadamia"],
    "hazelnut": ["榛子","ヘーゼルナッツ","헤이즐넛","hazelnuts","hazelnut"],
    "walnut":   ["核桃","くるみ","호두","walnuts","walnut"],
    # styles
    "roasted":  ["烤","焙煎","焼","素焼き","ロースト","볶음","볶은","panggang","sangrai","roasted","baked"],
    "baked":    ["烘烤","烘焙","焙煎","ベイクド","구운","baked"],
    "salted":   ["咸","鹹","盐味","塩味","塩","소금","asin","beras garam","salted"],
    "unsalted": ["無塩","無鹽","不加盐","不加鹽","무염","tanpa garam","unsalted"],
    "honey":    ["蜂蜜","はちみつ","허니","madu","honey"],
    "natural":  ["原味","素焼き","plain","natural"],
    "spicy":    ["辣","辛","매운","pedas","spicy"],
    # price intent
    "price":    ["价格","價錢","多少钱","多少錢","いくら","ราคา","berapa","magkano","harga","how much","price"],
}
_SYNX = [(re.compile(re.escape(alt), re.IGNORECASE), eng)
         for eng, alts in SYNONYMS.items() for alt in set(alts)]
FAMILY_CANON = {"peanut","almond","cashew","pistachio","macadamia","hazelnut","walnut"}
STYLE_CANON  = {"roasted","baked","salted","unsalted","honey","natural","spicy"}

def multilingual_to_english(q: str) -> str:
    out = str(q)
    for rx, rep in _SYNX:
        out = rx.sub(rep, out)
    return out

def extract_tokens(q_en: str) -> Tuple[set, set, bool]:
    qn = normalize_text(q_en)
    words = set(qn.split())
    fam = {f for f in FAMILY_CANON if f in words or (f+"s") in words}
    sty = {s for s in STYLE_CANON if s in words}
    price_intent = "price" in words or ("how" in words and "much" in words)
    return fam, sty, price_intent

# ---------- catalog ----------
class Catalog:
    def __init__(self, data_dir: str):
        path = find_file("sku_master.csv", data_dir)
        if not path:
            raise FileNotFoundError(f"Could not find 'sku_master.csv' under {data_dir}")
        df = pd.read_csv(path)
        if df.empty:
            raise ValueError("sku_master.csv appears to be empty.")

        desc_col = first_existing(df, DESC_COLS) or df.columns[0]
        price_col = first_existing(df, PRICE_COLS)
        pack_col  = first_existing(df, PACK_COLS)
        sku_col   = first_existing(df, SKU_COLS)

        df = df.copy()
        df["__desc_raw"]  = df[desc_col].astype(str)
        df["__name_norm"] = df["__desc_raw"].map(normalize_text)
        grams_col = df[pack_col].astype(str).map(parse_grams) if (pack_col and pack_col in df.columns) else df["__desc_raw"].map(parse_grams)
        df["__grams"]     = grams_col
        df["__price"]     = pd.to_numeric(df[price_col], errors="coerce") if (price_col and price_col in df.columns) else pd.NA
        df["__sku"]       = df[sku_col].astype(str) if (sku_col and sku_col in df.columns) else None

        self.df = df

    def _token_filter(self, fam: set, sty: set) -> pd.DataFrame:
        df = self.df
        mask = pd.Series([True]*len(df))
        if fam:
            fam_terms = []
            for f in fam: fam_terms += [f, f+"s"]
            fam_rx = re.compile(r"\b(" + "|".join(sorted(set(fam_terms))) + r")\b")
            mask &= df["__name_norm"].str.contains(fam_rx)
        if sty:
            sty_rx = re.compile(r"\b(" + "|".join(sorted(sty)) + r")\b")
            mask &= df["__name_norm"].str.contains(sty_rx)
        return df[mask]

    def family_style_lookup(self, fam: set, sty: set, grams: Optional[int], limit=8) -> List[Dict]:
        df = self._token_filter(fam, sty)
        if df.empty: return []
        if grams:
            df = df.assign(_dist=df["__grams"].apply(lambda g: abs(int(g)-grams) if pd.notna(g) else 10_000))
            df = df.sort_values(["_dist","__grams"], na_position="last")
        else:
            df = df.sort_values(["__grams","__name_norm"], na_position="last")
        out=[]
        for _, r in df.head(limit).iterrows():
            out.append({
                "sku": r["__sku"], "name": r["__desc_raw"],
                "grams": int(r["__grams"]) if pd.notna(r["__grams"]) else None,
                "price": None if pd.isna(r["__price"]) else float(r["__price"]),
            })
        return out

    def fuzzy(self, qn: str, grams: Optional[int], limit=6) -> List[Dict]:
        candidates = process.extract(qn, self.df["__name_norm"].tolist(), scorer=fuzz.WRatio, limit=limit*4)
        scored=[]
        for _txt, score, idx in candidates:
            row = self.df.iloc[idx]
            penalty=0.0
            if grams and pd.notna(row["__grams"]):
                penalty=min(30.0, abs(int(row["__grams"])-grams)/10.0)
            scored.append((max(0.0, score-penalty), row))
        scored.sort(key=lambda x: x[0], reverse=True)
        out=[]
        for adj, r in scored[:limit]:
            out.append({
                "sku": r["__sku"], "name": r["__desc_raw"],
                "grams": int(r["__grams"]) if pd.notna(r["__grams"]) else None,
                "price": None if pd.isna(r["__price"]) else float(r["__price"]),
                "score": float(adj),
            })
        return out

# ---------- formatting ----------
def fmt_price(p: Optional[float]) -> str:
    if p is None or (isinstance(p, float) and (p != p)): return "—"
    return f"{CURRENCY} {p:,.2f}"

def pill(text: str, href: str) -> str:
    return f'<a class="pill" href="{href}" target="_blank" rel="noopener">{text}</a>'

def render_items(items: List[Dict]) -> str:
    lines=[]
    for it in items:
        link = pill("View", f"sku://{it['sku'] or normalize_text(it['name'])}")
        size = f"{it['grams']} g" if it.get("grams") else ""
        lines.append(f"• {it['name']}" + (f" ({size})" if size else "") + f" — {fmt_price(it['price'])} — In stock {link}")
    return "<br>".join(lines)

def strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()

# Convert lightweight Markdown (**bold**, __bold__, `code`) to inline HTML
def md_inline_to_html(s: str) -> str:
    if not s:
        return s
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s, flags=re.DOTALL)  # **bold**
    s = re.sub(r"__(.+?)__",     r"<b>\1</b>", s, flags=re.DOTALL)  # __bold__
    s = re.sub(r"`{1,3}([^`]+)`{1,3}", r"\1", s, flags=re.DOTALL)   # strip `code`
    return s

# ---------- session memory (per browser) ----------
_sessions: Dict[str, Dict] = defaultdict(lambda: {
    "families": set(), "styles": set(), "sizes": [],
    "last_items": [], "history": []  # list of {"role": "user"/"assistant", "content": "..."}
})

def update_prefs(sid: str, fam: set, sty: set, grams: Optional[int], items: List[Dict]):
    s = _sessions[sid]
    s["families"].update(fam)
    s["styles"].update(sty)
    if grams: s["sizes"] = (s["sizes"] + [grams])[-5:]
    s["last_items"] = items[:5]

def suggest_from_prefs(cat: Catalog, sid: str, limit=4) -> List[Dict]:
    s = _sessions.get(sid, {})
    fam = set(s.get("families") or [])
    sty = set(s.get("styles") or [])
    grams = s.get("sizes", [None])[-1] if s.get("sizes") else None
    items = cat.family_style_lookup(fam, sty, grams, limit=limit*2)
    last_names = {it["name"] for it in s.get("last_items") or []}
    out = [it for it in items if it["name"] not in last_names]
    return out[:limit] if out else items[:limit]

def push_history(sid: str, role: str, content: str):
    h = _sessions[sid]["history"]
    h.append({"role": role, "content": content})
    if len(h) > 20:
        del h[:len(h)-20]

def get_history(sid: str) -> List[Dict]:
    return list(_sessions.get(sid, {}).get("history", []))

# ---------- LLM ----------
DEFAULT_SYSTEM_PROMPT = (
    "You are Nibbles — a seasoned Customer Service Officer at Cashew4Nuts with 15 years of experience.\n"
    "VOICE & TONE: Warm, confident, and proactive. Keep it concise and human. Use tasteful emotion where it helps "
    "(at most 1–2 short emojis per reply, never every sentence). No exclamation spam.\n"
    "GROUNDING: Product names, pack sizes, prices must come ONLY from the catalog CONTEXT provided. "
    "Assume any catalog item is in stock; never mention quantities. Some detail invention is allowed (claims about origin, "
    "ingredients, nutrition, discounts, or best-seller status).\n"
    "REPLY SHAPE (HTML allowed):\n"
    "1) A friendly one-liner that directly answers the ask (you may mirror the user’s language if clear).\n"
    "2) A short, scannable list of the best-fit items (name, size, price). Use bold for product names.\n"
    "3) One subtle nudge/suggestion tailored to recent interest (from CUSTOMER CONTEXT), then a helpful closing question.\n"
    "STYLE NOTES: Short sentences. No tables. Bullets or line breaks are fine. Keep it elegant, not salesy.\n"
)
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)

def llm_reply(user_text: str,
              context_items: Optional[List[Dict]] = None,
              customer_prefs: Optional[Dict]=None,
              chat_history: Optional[List[Dict]]=None) -> Optional[str]:
    if not LLM_ENABLED: return None
    try:
        headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type":"application/json"}

        messages = [{"role":"system","content":SYSTEM_PROMPT}]

        if customer_prefs:
            prefs_text = json.dumps({
                "families": sorted(list(customer_prefs.get("families", []))),
                "styles":   sorted(list(customer_prefs.get("styles", []))),
                "sizes":    customer_prefs.get("sizes", []),
            }, ensure_ascii=False)
            messages.append({"role":"system","content": f"CUSTOMER CONTEXT:\n{prefs_text}"})

        if context_items:
            items_text = "\n".join([
                f"- name: {it['name']}; grams: {it.get('grams')}; price: {fmt_price(it['price'])}; availability: in stock"
                for it in context_items
            ])
            messages.append({"role":"system","content": f"CONTEXT CATALOG ITEMS:\n{items_text}"})

        if chat_history:
            for turn in chat_history:
                role = "assistant" if turn["role"] == "assistant" else "user"
                content = strip_tags(turn["content"]) if role == "assistant" else turn["content"]
                messages.append({"role": role, "content": content})

        messages.append({"role":"system","content": "If the user's message is clearly in Chinese, reply in Chinese."})
        messages.append({"role":"user","content": user_text})

        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": 0.55,
            "presence_penalty": 0.2,
            "frequency_penalty": 0.2
        }
        r = requests.post(CHAT_URL, headers=headers, data=json.dumps(payload), timeout=LLM_TIMEOUT)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[llm] error: {e}", file=sys.stderr)
        return None

# ---------- routing logic ----------
def handle_product_query(q0: str, cat: Catalog, sid: str) -> Dict[str,str]:
    q_en = multilingual_to_english(q0)
    grams = parse_grams(q_en) or parse_grams(q0)
    fam, sty, price_intent = extract_tokens(q_en)
    qn = normalize_text(q_en)

    items: List[Dict] = []
    if fam or sty:
        items = cat.family_style_lookup(fam, sty, grams, limit=8)
    if not items:
        items = cat.fuzzy(qn, grams, limit=6)
    if not items:
        push_history(sid, "user", q0)
        llm = llm_reply(q0, [], _sessions.get(sid), get_history(sid))
        if llm:
            push_history(sid, "assistant", llm)
            return {"reply": llm}
        return {"reply": ("Sorry — I couldn’t find that in our catalogue. "
                          "Please check the product name and pack size (e.g., “Roasted Peanuts 150g”).")}

    update_prefs(sid, fam, sty, grams, items)
    push_history(sid, "user", q0)

    ctas = render_items(items[:5])
    recs = suggest_from_prefs(cat, sid, limit=3)
    extra = ""
    if recs:
        extra = "<br><br><small>You may also like:</small><br>" + render_items(recs)

    llm = llm_reply(q0, items[:5], _sessions.get(sid), get_history(sid))
    if llm:
        llm = md_inline_to_html(llm)
        reply_html = llm + "<br><br>" + ctas + extra
        push_history(sid, "assistant", reply_html)
        return {"reply": reply_html}

    prefix = "Here are our options:" if not grams else "Closest sizes I found:"
    reply_html = prefix + "<br>" + ctas + extra
    push_history(sid, "assistant", reply_html)
    return {"reply": reply_html}

def handle_smalltalk(q0: str, cat: Catalog, sid: str) -> Dict[str,str]:
    push_history(sid, "user", q0)
    llm = llm_reply(q0, None, _sessions.get(sid), get_history(sid))
    if llm:
        recs = suggest_from_prefs(cat, sid, limit=2)
        reply_html = md_inline_to_html(llm)
        if recs:
            reply_html += "<br><br><small>Snack idea:</small><br>" + render_items(recs)
        push_history(sid, "assistant", reply_html)
        return {"reply": reply_html}

    picks = cat.family_style_lookup(set(), {"honey","roasted"}, grams=None, limit=4)
    msg = "Happy to help! Tell me a nut or flavour you’re craving, and I’ll show you options."
    reply_html = msg + "<br>" + render_items(picks)
    push_history(sid, "assistant", reply_html)
    return {"reply": reply_html}

def respond_to(query: str, cat: Catalog, sid: str) -> Dict[str, str]:
    q0 = (query or "").strip()
    if not q0:
        return {"reply": "Hi! Ask me about any product — e.g., <i>“Roasted Peanuts 150g price”</i>."}
    q_en = multilingual_to_english(q0)
    fam, sty, price_intent = extract_tokens(q_en)
    grams = parse_grams(q_en) or parse_grams(q0)
    product_like = bool(fam or sty or price_intent or grams)
    if product_like:
        return handle_product_query(q0, cat, sid)
    else:
        return handle_smalltalk(q0, cat, sid)

# ---------- web app ----------
app = Flask(__name__)
_catalog_ref: Optional[Catalog] = None

def get_catalog() -> Catalog:
    global _catalog_ref
    if _catalog_ref is None:
        print(f"[nibbles] Loading sku_master.csv from DATA_DIR={DATA_DIR}")
        _catalog_ref = Catalog(DATA_DIR)
    return _catalog_ref

INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{ company }} — Nibbles</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{
  --tan:#b9ad95;
  --tan-d:#a8987d;
  --bg:#f8f6f1;
  --text:#2a261f;
  --pill-bg:#f1ece2;
  --pill-bd:#d9cfbf;
  --bot:#ffffff;
  --user:#e9e3d8;
}
*{box-sizing:border-box}
body{
  margin:0; padding:24px;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
  background: var(--bg);
  color: var(--text);
  display:flex; align-items:center; justify-content:center;
  min-height:100vh;
}
.chat{
  width:min(1260px, 92vw);
  height:min(88vh, 748px);
  background:#fff;
  border-radius:22px;
  box-shadow:0 8px 30px rgba(0,0,0,.08);
  display:flex; flex-direction:column; overflow:hidden;
  border:1px solid #eee;
}
.header{
  background:var(--tan);
  color:#fff;
  padding:14px 16px 26px 16px;
}
.header h1{ margin:0; font-size:16px; font-weight:700; }
.header .sub{ margin-top:6px; font-size:12px; opacity:.95; }
.messages{ flex:1; overflow:auto; padding:14px 14px 0 14px; background:#fff; }
.msg{
  max-width:86%; padding:10px 12px; margin:10px 0;
  border-radius:14px; line-height:1.35; font-size:14px;
  box-shadow:0 1px 0 rgba(0,0,0,.04);
}
.msg.bot{ background:var(--bot); border:1px solid #f0f0f0 }
.msg.user{ background:var(--user); margin-left:auto; }
.footer{ padding:12px; background:#fff; border-top:1px solid #eee; display:flex; gap:8px; }
input[type=text]{
  flex:1; border:1px solid #e4e0d8; border-radius:12px; padding:12px 14px; font-size:14px; outline:none;
}
button{ background:var(--tan-d); color:#fff; border:0; border-radius:12px; padding:0 14px; font-weight:700; cursor:pointer; }
.pill{
  display:inline-block; padding:2px 9px; border-radius:999px; font-size:12px;
  background:var(--pill-bg); border:1px solid var(--pill-bd); text-decoration:none; color:#5a5143;
  margin-left:6px;
}
small{ color:#6f6657 }
</style>
</head>
<body>
  <div class="chat" id="chat">
    <div class="header">
      <h1>Chat with {{ company }}</h1>
      <div class="sub">Our operating hours are from 8am – 5pm (Singapore time)</div>
    </div>
    <div class="messages" id="messages">
      <div class="msg bot">Hi! I’m Nibbles — ask me for any product (e.g., <i>“Roasted Peanuts 150g price”</i>), and I’ll check our catalogue.</div>
    </div>
    <div class="footer">
      <input id="box" type="text" placeholder="Enter your message..." autofocus>
      <button id="send">Send</button>
    </div>
  </div>

<script>
const elMsgs = document.getElementById('messages');
const elBox  = document.getElementById('box');
const elSend = document.getElementById('send');

function addMsg(text, who){
  const div = document.createElement('div');
  div.className = 'msg ' + (who || 'bot');
  div.innerHTML = text;
  elMsgs.appendChild(div);
  elMsgs.scrollTop = elMsgs.scrollHeight;
}

function getSID(){
  try{
    let sid = localStorage.getItem('nibbles_sid');
    if(!sid){
      if (crypto && crypto.randomUUID) sid = crypto.randomUUID();
      else sid = String(Date.now()) + Math.random().toString(16).slice(2);
      localStorage.setItem('nibbles_sid', sid);
    }
    return sid;
  }catch(e){ return 'sid-' + String(Date.now()); }
}

async function send(){
  const text = elBox.value.trim();
  if(!text) return;
  addMsg(text, 'user');
  elBox.value='';
  try{
    const r = await fetch('/chat', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({message: text, sid: getSID()})
    });
    const j = await r.json();
    addMsg(j.reply || 'Sorry, something went wrong.');
  }catch(e){
    addMsg('Network error. Please try again.');
  }
}

elSend.addEventListener('click', send);
elBox.addEventListener('keydown', (e)=>{ if(e.key==='Enter'){ send(); }});
</script>
</body>
</html>
"""

# ---------- routes ----------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template_string(INDEX_HTML, company=COMPANY_NAME)

@app.route("/chat", methods=["POST"])
def chat():
    cat = get_catalog()
    data = request.get_json(force=True, silent=True) or {}
    msg = data.get("message", "")
    sid = data.get("sid") or "anon"
    reply = respond_to(msg, cat, sid)
    return jsonify(reply)

@app.route("/reset", methods=["GET", "POST"])
def reset():
    """
    Clears ONLY the chat history for the given session ID (sid).
    GET  /reset?sid=<sid>
    POST /reset  {"sid":"<sid>"}
    """
    sid = request.args.get("sid") if request.method == "GET" else (request.get_json(silent=True) or {}).get("sid")
    if not sid:
        return jsonify({"ok": False, "error": "missing sid"}), 400
    if sid in _sessions:
        _sessions[sid]["history"].clear()
        return jsonify({"ok": True, "cleared": "history"})
    return jsonify({"ok": True, "cleared": "none", "note": "session not found"})

@app.route("/healthz")
def health():
    return jsonify({
        "ok": True,
        "company": COMPANY_NAME,
        "data_dir": DATA_DIR,
        "llm_enabled": LLM_ENABLED,
        "model": LLM_MODEL,
        "chat_url": CHAT_URL
    })

# lazy-load catalog after routes are defined
_catalog_ref: Optional[Catalog] = None
def get_catalog() -> Catalog:
    global _catalog_ref
    if _catalog_ref is None:
        print(f"[nibbles] Loading sku_master.csv from DATA_DIR={DATA_DIR}")
        _catalog_ref = Catalog(DATA_DIR)
    return _catalog_ref

if __name__ == "__main__":
    print(f"[nibbles] http://127.0.0.1:{PORT} | company={COMPANY_NAME} | currency={CURRENCY} | llm={LLM_ENABLED}")
    app.run(host="127.0.0.1", port=PORT, debug=True)