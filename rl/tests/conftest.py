import importlib.util
from pathlib import Path

# Reuse the engine's tiny checkpoint fixtures (loaded under another module name:
# both files are called conftest).
_spec = importlib.util.spec_from_file_location(
    "engine_conftest", Path(__file__).resolve().parents[2] / "engine" / "tests" / "conftest.py"
)
_engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_engine)
tiny_qwen3 = _engine.tiny_qwen3
