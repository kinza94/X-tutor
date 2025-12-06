# app.py

import os
import sys
import json
import time
import datetime
import re
import random
import math
import hashlib
from io import BytesIO
from fractions import Fraction
from typing import List, Dict, Optional

# Attempt rich libraries (optional)
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except Exception:
    STREAMLIT_AVAILABLE = False

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except Exception:
    SYMPY_AVAILABLE = False

try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except Exception:
    PYPDF2_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except Exception:
    PDFPLUMBER_AVAILABLE = False

try:
    import docx
    from docx.shared import Pt
    DOCX_AVAILABLE = True
except Exception:
    DOCX_AVAILABLE = False

# OpenAI optional
OPENAI_AVAILABLE = False
openai_client_v1 = None
openai_module_old = None
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "") or ""
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4o-mini")
if OPENAI_API_KEY:
    try:
        from openai import OpenAI as _OpenAI  # type: ignore
        openai_client_v1 = _OpenAI(api_key=OPENAI_API_KEY)
        OPENAI_AVAILABLE = True
    except Exception:
        try:
            import openai as _openai_old  # type: ignore
            _openai_old.api_key = OPENAI_API_KEY
            openai_module_old = _openai_old
            OPENAI_AVAILABLE = True
        except Exception:
            OPENAI_AVAILABLE = False

# Logging
LOGFILE = "x_tutor_log.jsonl"
def log_query(entry: dict):
    try:
        with open(LOGFILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception:
        pass

# ---------------------------
# Plugin bootstrapping
# ---------------------------
TOPICS_DIR = os.path.join(os.path.dirname(__file__), "topics")

ALGEBRA_LINEAR_PLUGIN = r'''# topics/algebra_linear.py
import re
import sympy as sp

name = "Algebra — Linear"
tags = ["algebra", "linear", "equation"]

def can_handle(question: str) -> bool:
    # We'll generally let the main app handle simple linear equations,
    # so we keep this conservative.
    q = question.strip().lower()
    if "=" not in q:
        return False
    # Avoid advanced stuff in this plugin; leave to main solver.
    if any(tok in q for tok in ["sin(", "cos(", "integral", "d/d", "log("]):
        return False
    # This plugin can be used as a fallback if needed.
    return False  # effectively disabled; main app handles linear
'''

QUADRATIC_PLUGIN = r'''# topics/quadratic.py
import re
import sympy as sp

name = "Algebra — Quadratic"
tags = ["algebra", "quadratic", "equation"]

def can_handle(question: str) -> bool:
    q = question.strip().lower()
    if "=" not in q:
        return False
    return bool(re.search(r"\b(x|[a-zA-Z])\s*\^?2\b", q)) or "^2" in q or "squared" in q

def explain(question: str, level: str = "Intermediate", context: str = ""):
    try:
        q = question.strip()
        if q.lower().startswith("solve "):
            q = q.split(None,1)[1]
        left, right = q.split("=",1)
        left_e = sp.sympify(left)
        right_e = sp.sympify(right)
        syms = list(left_e.free_symbols.union(right_e.free_symbols))
        if not syms:
            return {"answer": f"{sp.N(left_e)} vs {sp.N(right_e)}",
                    "steps":[f"{left} = {right}",
                             f"Left = {sp.N(left_e)}",
                             f"Right = {sp.N(right_e)}"],
                    "meta":{"key_idea":"numeric"}}
        var = syms[0]
        sol = sp.solve(sp.Eq(left_e, right_e), var)
        steps = [f"Equation: {left} = {right}",
                 f"Variable: {var}",
                 f"Roots: {sol}"]
        return {"answer": f"{var} = {sol}",
                "steps": steps,
                "meta": {"key_idea":"quadratic formula"}}
    except Exception as e:
        return {"answer": f"Parse error: {e}", "steps": [str(e)], "meta": {}}
'''

def ensure_topics_folder():
    os.makedirs(TOPICS_DIR, exist_ok=True)
    init = os.path.join(TOPICS_DIR, "__init__.py")
    if not os.path.exists(init):
        with open(init, "w", encoding="utf-8") as f:
            f.write("# topics package\n")
    linear_path = os.path.join(TOPICS_DIR, "algebra_linear.py")
    if not os.path.exists(linear_path):
        with open(linear_path, "w", encoding="utf-8") as f:
            f.write(ALGEBRA_LINEAR_PLUGIN)
    quad_path = os.path.join(TOPICS_DIR, "quadratic.py")
    if not os.path.exists(quad_path):
        with open(quad_path, "w", encoding="utf-8") as f:
            f.write(QUADRATIC_PLUGIN)

# Ensure topics/plugins exist
ensure_topics_folder()

# ---------------------------
# Plugin loader & router
# ---------------------------
import importlib
import pkgutil

def load_plugins(package_name: str = "topics"):
    plugins = []
    try:
        pkg = importlib.import_module(package_name)
    except Exception:
        return plugins
    for finder, modname, ispkg in pkgutil.iter_modules(pkg.__path__):
        if ispkg:
            continue
        try:
            module = importlib.import_module(f"{package_name}.{modname}")
            if hasattr(module, "can_handle") and hasattr(module, "explain") and hasattr(module, "name"):
                plugins.append(module)
        except Exception:
            continue
    return plugins

def route_question(question: str, plugins: List, preferred_topic: Optional[str] = None):
    """
    Route only *non-simple-linear* things to plugins.
    Linear equations are handled by our own detailed solver.
    """
    q = question.strip().lower()

    # if it's a simple linear equation (one variable, no ^2 etc.), skip plugins
    if "=" in q and re.search(r"[a-zA-Z]", q) and "^" not in q and "squared" not in q and "²" not in q:
        return None

    if preferred_topic and preferred_topic != "Auto-detect":
        for p in plugins:
            if preferred_topic.lower() in getattr(p, "name", "").lower() or \
               preferred_topic.lower() in getattr(p, "tags", []):
                return p
    candidates = [p for p in plugins if p.can_handle(question)]
    return candidates[0] if candidates else None

PLUGINS = load_plugins()

# ---------------------------
# Local solver helpers
# ---------------------------
def _split_terms(expr: str):
    s = expr.replace(" ", "")
    if s and s[0] not in "+-":
        s = "+" + s
    return re.findall(r"[+-][^+-]+", s)

def _coeff_and_const_from_expr(expr: str, var: str):
    coef = Fraction(0, 1)
    const = Fraction(0, 1)
    terms = _split_terms(expr)
    for t in terms:
        if var in t:
            part = t.replace(var, "").replace("*", "")
            if part in ["+", ""]:
                c = Fraction(1, 1)
            elif part == "-":
                c = Fraction(-1, 1)
            else:
                try:
                    c = Fraction(part)
                except Exception:
                    c = Fraction(float(part))
            coef += c
        else:
            try:
                const += Fraction(t)
            except Exception:
                const += Fraction(float(t))
    return coef, const

def _find_variable(expr1: str, expr2: str):
    m = re.search(r"[a-zA-Z]", expr1)
    if m:
        return m.group(0)
    m = re.search(r"[a-zA-Z]", expr2)
    if m:
        return m.group(0)
    return None

# ---------------------------
# Simple NLP: text → equation (very basic)
# ---------------------------
def text_to_equation(text: str, var: str = "x") -> Optional[str]:
    """
    Very simple rule-based NLP:
    Handles sentences like:
      - "a number plus 5 is 12"
      - "5 more than a number is 9"
      - "twice a number plus 3 is 11"
      - "5 less than a number is 10"
    Returns algebraic equation string or None.
    """
    t = text.lower().strip()

    # If user already gave equation, don't touch
    if "=" in t:
        return None

    nums = re.findall(r"-?\d+", t)
    nums = [int(n) for n in nums] if nums else []
    if not nums:
        return None

    coef = 1
    if "twice" in t or "2 times" in t or "two times" in t:
        coef = 2
    elif "thrice" in t or "3 times" in t or "three times" in t:
        coef = 3

    # plus / more than / increased by
    if "number" in t and any(k in t for k in ["plus", "more than", "add", "increased by", "sum of"]):
        if len(nums) >= 2:
            b, c = nums[-2], nums[-1]  # last two numbers
            return f"{coef}{var} + {b} = {c}"

    # minus / less than / decreased by
    if "number" in t and any(k in t for k in ["less than", "minus", "decreased by", "subtracted from"]):
        if len(nums) >= 2:
            b, c = nums[-2], nums[-1]
            # assume: "b less than a number is c" -> x - b = c
            return f"{coef}{var} - {b} = {c}"

    # "twice a number is 14"
    if "number" in t and any(k in t for k in ["times", "multiplied by"]) and \
       "plus" not in t and "minus" not in t:
        c = nums[-1]
        return f"{coef}{var} = {c}"

    return None

# ---------------------------
# Local linear equation solver with detailed steps
# ---------------------------
def solve_equation_local(question: str):
    """
    Detailed step-by-step solver for linear equations.
    Also handles numeric evaluation and simple NLP word problems.
    """
    q = question.strip()
    if not q:
        return None

    # Allow "solve x+2=5" style
    if q.lower().startswith("solve"):
        parts = q.split(None, 1)
        if len(parts) > 1:
            q = parts[1].strip()

    # First try to convert natural language -> equation
    nlp_eq = text_to_equation(q)
    if nlp_eq is not None:
        q = nlp_eq

    # If still no "=", try numeric expression
    if "=" not in q:
        try:
            if SYMPY_AVAILABLE:
                v = sp.N(sp.sympify(q))
            else:
                v = eval(q, {"__builtins__": None}, {})
            return {"answer": str(v), "steps": [f"Evaluate {q} = {v}"]}
        except Exception:
            return None

    left, right = q.split("=", 1)
    left = left.strip()
    right = right.strip()

    var = _find_variable(left, right)
    if not var:
        # maybe just numeric equality
        try:
            lval = Fraction(left)
            rval = Fraction(right)
            steps = [f"{left} = {right}", f"Left = {lval}", f"Right = {rval}"]
            return {
                "answer": "True" if lval == rval else "False",
                "steps": steps,
            }
        except Exception:
            return None

    # get coefficients on both sides
    try:
        coef_l, const_l = _coeff_and_const_from_expr(left, var)
        coef_r, const_r = _coeff_and_const_from_expr(right, var)
    except Exception:
        return None

    # a x + c = 0 form  ->  a = coef_l - coef_r,  c = const_l - const_r
    a = coef_l - coef_r          # combined x coefficient
    c = const_l - const_r        # combined constant on LHS
    N = -c                       # RHS constant after moving

    steps: List[str] = []

    # Step-by-step explanation
    steps.append(f"Step 1: Start with the equation: {left} = {right}")
    steps.append(
        f"Step 2: Combine like terms on each side: "
        f"{coef_l}{var} + {const_l} = {coef_r}{var} + {const_r}"
    )
    steps.append(
        f"Step 3: Move x-terms to the left and constants to the right: "
        f"({coef_l} - {coef_r}){var} = {const_r} - {const_l}"
    )
    steps.append(f"Step 4: Simplify: {a}{var} = {N}")

    # special cases
    if a == 0:
        if c == 0:
            steps.append("Both sides reduce to 0 = 0 → infinitely many solutions.")
            return {"answer": f"Infinite solutions (any {var})", "steps": steps}
        else:
            steps.append("x-terms cancel but constants differ → no solution.")
            return {"answer": "No solution", "steps": steps}

    # normal case
    steps.append(f"Step 5: Divide both sides by {a}: {var} = {N}/{a}")

    # Factor breakdown if integer division possible
    if (
        isinstance(a, Fraction)
        and isinstance(N, Fraction)
        and a.denominator == 1
        and N.denominator == 1
    ):
        Ai = a.numerator
        Ni = N.numerator
        if Ai != 0 and Ni % Ai == 0:
            k = Ni // Ai
            steps.append("Factor breakdown:")
            steps.append(f"{Ni} → {Ai} × {k}")
            steps.append(f"{Ai} → {Ai} × 1")
            steps.append(f"So {var} = {k}")
            return {"answer": f"{var} = {k}", "steps": steps}

    # otherwise leave as fraction
    sol = N / a
    steps.append(f"Simplify: {var} = {sol}")
    return {"answer": f"{var} = {sol}", "steps": steps}

# ---------------------------
# PDF helpers (best-effort)
# ---------------------------
def load_pdf_pages_from_fileobj(file_obj):
    pages = []
    try:
        file_obj.seek(0)
    except Exception:
        pass
    if PDFPLUMBER_AVAILABLE:
        try:
            with pdfplumber.open(file_obj) as pdf:
                for p in pdf.pages:
                    try:
                        txt = p.extract_text() or ""
                    except Exception:
                        txt = ""
                    pages.append(txt)
            if any(pages):
                return pages
        except Exception:
            pass
    if PYPDF2_AVAILABLE:
        try:
            reader = PyPDF2.PdfReader(file_obj)
            for p in reader.pages:
                try:
                    txt = p.extract_text() or ""
                except Exception:
                    txt = ""
                pages.append(txt)
            if any(pages):
                return pages
        except Exception:
            pass
    return pages

# ---------------------------
# Worksheet generator
# ---------------------------
def generate_questions(topic, n):
    """Generate worksheet questions based on selected topic."""
    if topic == "Ratio":
        return generate_ratio_questions(n)

    elif topic == "Multiplication":
        return [f"{random.randint(2,12)} × {random.randint(2,12)} = ?" for _ in range(n)]

    elif topic == "Fractions":
        qs = []
        for _ in range(n):
            a, b = random.randint(1,9), random.randint(2,9)
            c, d = random.randint(1,9), random.randint(2,9)
            qs.append(f"{a}/{b} + {c}/{d} = ?")
        return qs

    elif topic == "Percentage":
        qs = []
        for _ in range(n):
            num = random.randint(50, 500)
            perc = random.choice([5,10,15,20,25,30])
            qs.append(f"Find {perc}% of {num}.")
        return qs

    elif topic == "Word Problems":
        return [
            "Ali has 5 pencils. He buys 7 more. How many pencils now?",
            "Sara had 20 mangoes. She gave 6 away. How many mangoes left?",
            "A bag has 45 candies. 15 were eaten. How many remain?",
        ][:n]

    # fallback
    return [f"Question {i+1}" for i in range(n)]

def generate_ratio_questions(n):
    qs = []
    examples = [(16, 32), (160, 200), (12, 18), (5, 20), (9, 12)]

    for a, b in examples:
        if len(qs) < n:
            qs.append(f"Find the ratio of {a} and {b}.")

    while len(qs) < n:
        a, b = random.randint(2, 200), random.randint(2, 200)
        qs.append(f"Find the ratio of {a} and {b}.")

    return qs

def simplify_ratio_text(q_text):
    nums = re.findall(r"-?\d+", q_text)
    if len(nums) >= 2:
        a, b = int(nums[0]), int(nums[1])

        if a == 0 and b == 0:
            return "Undefined (both zero)"

        g = math.gcd(a, b)
        return f"{a//g} : {b//g}"

    return "Could not parse numbers."

# ---------------------------
# Simple OpenAI wrapper (optional)
# ---------------------------
def call_gpt_simple(prompt: str, max_tokens:int=400, temperature:float=0.0):
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI not configured")
    if openai_client_v1 is not None:
        resp = openai_client_v1.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role":"system","content":"You are X-Tutor."},
                      {"role":"user","content":prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        try:
            return resp.choices[0].message.content
        except Exception:
            return str(resp)
    elif openai_module_old is not None:
        resp = openai_module_old.ChatCompletion.create(
            model=LLM_MODEL,
            messages=[{"role":"system","content":"You are X-Tutor."},
                      {"role":"user","content":prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        try:
            return resp.choices[0].text
        except Exception:
            return str(resp)
    else:
        raise RuntimeError("OpenAI client missing")

# Prompt builder for LLM tutor (you can hook this in UI later)
def build_tutor_prompt(question: str, level: str, context_text: str = "") -> str:
    """
    Prompt that turns the LLM into a step-by-step tutor.
    """
    prompt = f"""
You are X-Tutor, a friendly {level.lower()}-level teacher.

Student question:
{question}

{("Here is relevant textbook content:\n" + context_text) if context_text else ""}

Your job:
1. Restate the question in simple words.
2. Briefly explain the key concept.
3. Solve step-by-step (number your steps).
4. At the end, clearly state:
   - Final answer
   - One common mistake students make
   - One short tip to remember the concept.

Use clear headings and short sentences.
If there is math, show it neatly.
"""
    return prompt.strip()

# ---------------------------
# UI: Streamlit
# ---------------------------
def run_streamlit():
    st.set_page_config(page_title="X-Tutor — Modular", layout="wide")
    st.title("🧠 X-Tutor — Modular & Extensible")

    # Sidebar
    with st.sidebar:
        st.header("Settings")
        uploaded_pdfs = st.file_uploader("Upload PDFs (multiple allowed)", type=['pdf'], accept_multiple_files=True)
        subject = st.selectbox("Subject", ["Mathematics","Physics","Chemistry"], index=0)
        level = st.radio("Explain level", ["Beginner","Intermediate","Advanced"], index=1)
        show_meta = st.checkbox("Show meta/notes", value=True)
        use_gpt_ws = st.checkbox("Use GPT for worksheet generation (tokens)", value=False)
        use_llm_tutor = st.checkbox("Use LLM Tutor (if API key set)", value=False)
        debug_mode = st.checkbox("Debug info", value=False)
        st.markdown("---")
        if uploaded_pdfs:
            if st.button("Save uploaded PDFs to textbooks/"):
                saved = 0
                for f in uploaded_pdfs:
                    try:
                        os.makedirs("textbooks", exist_ok=True)
                        dest = os.path.join("textbooks", getattr(f, "name", f"uploaded_{int(time.time())}.pdf"))
                        with open(dest, "wb") as out:
                            out.write(f.getbuffer())
                        saved += 1
                    except Exception:
                        pass
                st.success(f"Saved {saved} files to textbooks/")

    # Load PDF list
    pdf_docs = []
    if os.path.exists("textbooks"):
        for fname in sorted(os.listdir("textbooks")):
            if fname.lower().endswith(".pdf"):
                try:
                    with open(os.path.join("textbooks", fname), "rb") as fh:
                        pages = load_pdf_pages_from_fileobj(fh)
                        pdf_docs.append({"name": fname, "pages": pages})
                except Exception:
                    pass

    left_col, right_col = st.columns([2,1])
    with left_col:
        question = st.text_area(
            "Type your question here:",
            height=160,
            placeholder="e.g. x+2=5 or 2x+5=20 or 'a number plus 3 is 10'"
        )
        st.markdown("### Worksheet generator")

        ws_topic = st.selectbox(
            "Worksheet topic",
            ["Ratio", "Multiplication", "Fractions", "Percentage", "Word Problems"]
        )

        ws_n = st.number_input("Number of questions", min_value=5, max_value=50, value=12)

        if st.button("Preview worksheet"):
            qs = generate_questions(ws_topic, ws_n)
            st.markdown("### Preview:")
            for i, q in enumerate(qs, start=1):
                st.write(f"{i}. {q}")

        use_context = st.checkbox("Use selected textbook page as context", value=True)
        if pdf_docs:
            selected_pdf_idx = st.selectbox(
                "Select textbook",
                options=list(range(len(pdf_docs))),
                format_func=lambda i: pdf_docs[i]["name"]
            )
            pages = pdf_docs[selected_pdf_idx]["pages"]
            if pages:
                selected_page_num = st.selectbox(
                    "Select page",
                    options=list(range(1, len(pages)+1)),
                    index=0
                )
                page_text = pages[selected_page_num-1]
                st.write("Page preview:")
                st.write(page_text[:1000] + ("..." if len(page_text)>1000 else ""))
            else:
                selected_page_num = None
        else:
            selected_pdf_idx = None
            selected_page_num = None

        col_run, col_clear = st.columns([1,1])
        if col_clear.button("Clear history"):
            st.session_state["history"] = []

        if col_run.button("Get Explanation"):
            if not question or not question.strip():
                st.error("Type a question.")
            else:
                context_text = ""
                if use_context and selected_pdf_idx is not None and selected_page_num is not None:
                    context_text = pdf_docs[selected_pdf_idx]["pages"][selected_page_num-1]

                # 1) Try plugin (e.g., quadratic)
                plugin = route_question(question, PLUGINS)

                if plugin is None:
                    # Local linear / numeric solver
                    fb = solve_equation_local(question)
                    if fb:
                        st.subheader("Final Answer (Local Solver)")
                        st.success(fb.get("answer", ""))
                        st.subheader("Steps")
                        for s in fb.get("steps", []):
                            st.write("-", s)
                    else:
                        st.error("No plugin matched and no local solution available.")
                else:
                    try:
                        res = plugin.explain(question, level=level, context=context_text)
                        st.subheader("Final Answer (Plugin)")
                        st.success(res.get("answer",""))
                        st.subheader("Step-by-step Explanation")
                        for s in res.get("steps",[]):
                            st.write("-", s)
                        if show_meta and res.get("meta"):
                            st.subheader("Notes")
                            for k,v in res.get("meta",{}).items():
                                st.write(f"- {k}: {v}")
                    except Exception as e:
                        st.error(f"Plugin error: {e}")

                # 2) Optional LLM tutor explanation (separate section)
                if use_llm_tutor:
                    if not OPENAI_AVAILABLE:
                        st.warning("OPENAI_API_KEY not set. LLM Tutor disabled.")
                    else:
                        try:
                            tutor_prompt = build_tutor_prompt(question, level, context_text)
                            llm_answer = call_gpt_simple(tutor_prompt, max_tokens=700, temperature=0.2)
                            st.markdown("---")
                            st.subheader("🧠 X-Tutor (LLM) — Answer & Explanation")
                            st.write(llm_answer)
                        except Exception as e:
                            st.error(f"LLM Tutor error: {e}")

    with right_col:
        st.markdown("### Recent Q&A")
        history = st.session_state.get("history", [])
        if not history:
            st.write("No recent Q&A.")
        else:
            for h in history[:10]:
                with st.expander(f"{h['timestamp']} — {h['question'][:60]}"):
                    st.write(h['answer'])
        st.markdown("---")
        st.markdown("### Upload CSV of questions (optional)")
        uploaded = st.file_uploader("Upload CSV with column 'question'", type=['csv'])
        if uploaded is not None:
            try:
                import pandas as pd
                df = pd.read_csv(uploaded)
                if 'question' not in df.columns:
                    st.error("CSV must contain column named 'question'.")
                else:
                    st.dataframe(df.head())
                    if st.button("Run batch explanations"):
                        results = []
                        for q in df['question'].astype(str).tolist():
                            fb = solve_equation_local(q)
                            answers = fb.get("answer","") if fb else "No local answer"
                            results.append({'question': q, 'answer': answers})
                        res_df = pd.DataFrame(results)
                        csv_out = res_df.to_csv(index=False)
                        st.download_button(
                            "Download results CSV",
                            data=csv_out,
                            file_name='batch_results.csv',
                            mime='text/csv'
                        )
            except Exception as e:
                st.error(f"Error reading CSV: {e}")

    st.markdown("---")
    c1, c2 = st.columns([3,1])
    with c1:
        st.write("Study with ❤️ — X-Tutor")
    with c2:
        st.write("v1.0")

# ---------------------------
# CLI fallback
# ---------------------------
def run_cli():
    print("X-Tutor — CLI mode (local solver)")
    print(f"SymPy available: {SYMPY_AVAILABLE}")
    print("Examples: x+2=5   2x+5=20   'a number plus 3 is 10'")
    print("Type 'exit' or Ctrl+C to quit.")
    while True:
        try:
            q = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return
        if not q:
            continue
        if q.lower() in ("exit","quit"):
            print("Goodbye!")
            return

        plugin = route_question(q, PLUGINS)
        if plugin:
            try:
                res = plugin.explain(q, level="Intermediate", context="")
                print("Answer:", res.get("answer",""))
                print("Steps:")
                for s in res.get("steps",[]): print("-", s)
            except Exception as e:
                print("Plugin error:", e)
                fb = solve_equation_local(q)
                if fb:
                    print("Local answer:", fb.get("answer",""))
        else:
            fb = solve_equation_local(q)
            if fb:
                print("Answer:", fb.get("answer",""))
                print("Steps:")
                for s in fb.get("steps",[]): print("-", s)
            else:
                print("No local solution available. Install SymPy or add plugin.")

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    if STREAMLIT_AVAILABLE:
        run_streamlit()
    else:
        print("Streamlit not installed — running CLI fallback. To use web UI: pip install streamlit")
        run_cli()
