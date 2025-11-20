# core/solver.py
import re
from fractions import Fraction
try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except Exception:
    SYMPY_AVAILABLE = False

def _split_terms(expr: str):
    s = expr.replace(' ', '')
    if s and s[0] not in '+-':
        s = '+' + s
    return re.findall(r'[+-][^+-]+', s)

def _coeff_and_const_from_expr(expr: str, var: str):
    coef = Fraction(0,1)
    const = Fraction(0,1)
    terms = _split_terms(expr)
    for t in terms:
        if var in t:
            part = t.replace(var, '').replace('*','')
            if part in ('','+'):
                c = Fraction(1,1)
            elif part == '-':
                c = Fraction(-1,1)
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
    m = re.search(r'[a-zA-Z]', expr1)
    if m: return m.group(0)
    m = re.search(r'[a-zA-Z]', expr2)
    if m: return m.group(0)
    return None

def solve_equation_local(question: str):
    q = question.strip()
    if not q: return None
    if q.lower().startswith('solve '):
        q = q.split(None,1)[1]
    if '=' not in q:
        try:
            if SYMPY_AVAILABLE:
                v = sp.N(sp.sympify(q))
                return {"answer": str(v), "steps":[f"Evaluate: {q} = {v}"]}
            else:
                v = eval(q, {"__builtins__": None}, {})
                return {"answer": str(v), "steps":[f"Evaluate: {q} = {v}"]}
        except Exception:
            return None
    left, right = q.split('=',1)
    left = left.strip(); right = right.strip()
    if SYMPY_AVAILABLE:
        try:
            left_e = sp.sympify(left); right_e = sp.sympify(right)
            syms = list(left_e.free_symbols.union(right_e.free_symbols))
            if not syms:
                lval = sp.N(left_e); rval = sp.N(right_e)
                return {"answer": "True" if sp.Eq(lval,rval) else "False",
                        "steps":[f"{left} = {right}", f"Left = {lval}", f"Right = {rval}"]}
            var = syms[0]
            sol = sp.solve(sp.Eq(left_e, right_e), var)
            return {"answer": f"{var} = {sol}", "steps":[f"Solve: {left} = {right}", f"Variable: {var}", f"Solution: {sol}"]}
        except Exception:
            pass
    var = _find_variable(left, right)
    if not var:
        try:
            lval = Fraction(left); rval = Fraction(right)
            return {"answer": "True" if lval==rval else "False", "steps":[f"{left} = {right}", f"Left = {lval}", f"Right = {rval}"]}
        except Exception:
            return None
    try:
        coef_l, const_l = _coeff_and_const_from_expr(left, var)
        coef_r, const_r = _coeff_and_const_from_expr(right, var)
    except Exception:
        return None
    a = coef_l - coef_r
    c = const_l - const_r
    if a == 0:
        if c == 0:
            return {"answer": f"Infinite solutions (any {var})", "steps":[f"0 = 0 → infinite solutions"]}
        else:
            return {"answer":"No solution", "steps":[f"Contradiction: {c} = 0"]}
    sol = -c / a
    steps = [f"Start: {left} = {right}", f"Identify: {var}", f"{a}*{var} + ({c}) = 0", f"{var} = {sol}"]
    return {"answer": f"{var} = {sol}", "steps": steps}
