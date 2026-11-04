"""Entry point: `streamlit run app.py`. The dashboard itself lives in scripts/demo_app.py (one copy, no duplication)."""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).parent / "scripts" / "demo_app.py"), run_name="__main__")
