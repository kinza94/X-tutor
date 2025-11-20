# topics/quadratic.py
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
            return {"answer": f"{sp.N(left_e)} vs {sp.N(right_e)}", "steps":[f"{left} = {right}", f"Left = {sp.N(left_e)}", f"Right = {sp.N(right_e)}"], "meta":{"key_idea":"numeric"}}
        var = syms[0]
        sol = sp.solve(sp.Eq(left_e, right_e), var)
        steps = [f"Equation: {left} = {right}", f"Variable: {var}", f"Roots: {sol}"]
        return {"answer": f"{var} = {sol}", "steps": steps, "meta": {"key_idea":"quadratic formula"}}
    except Exception as e:
        return {"answer": f"Parse error: {e}", "steps": [str(e)], "meta": {}}
