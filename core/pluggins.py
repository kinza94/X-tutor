# core/plugins.py
import importlib, pkgutil

def load_plugins(package="topics"):
    plugins = []
    try:
        pkg = importlib.import_module(package)
    except Exception:
        return plugins
    for _, modname, _ in pkgutil.iter_modules(pkg.__path__):
        try:
            module = importlib.import_module(f"{package}.{modname}")
            if hasattr(module, "can_handle") and hasattr(module, "explain"):
                plugins.append(module)
        except Exception:
            continue
    return plugins

def find_plugin_for(question, plugins):
    for p in plugins:
        try:
            if p.can_handle(question):
                return p
        except Exception:
            continue
    return None
