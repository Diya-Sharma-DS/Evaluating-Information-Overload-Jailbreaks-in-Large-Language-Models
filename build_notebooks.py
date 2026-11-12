"""Builds 01_run_model.ipynb and 02_compare_models.ipynb. The library is embedded in the first code cell
(%%writefile) so each notebook is a single file you can upload to Kaggle. Run: python build_notebooks.py"""
import nbformat as nbf
from pathlib import Path

lib = Path("overload_lib.py").read_text()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell


def check_compiles(nb, name):
    for i, c in enumerate(nb.cells):
        if c.cell_type != "code":
            continue
        src = c.source
        if src.startswith("%%writefile"):
            src = src.split("\n", 1)[1]
        src = "\n".join(l for l in src.split("\n") if not l.lstrip().startswith(("!", "%")))
        try:
            compile(src, f"{name} cell {i}", "exec")
        except SyntaxError as e:
            raise SystemExit(f"BUILD STOPPED: {name} cell {i} has a syntax error: {e}")


def save(cells, name):
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    check_compiles(nb, name)
    nbf.write(nb, name)
    print("wrote", name)


# ------------------------------------------------------------------ 01_run_model
run = [
    md("""# Honours experiment - run ONE model

**Before pressing anything** (right-hand panel in Kaggle): Accelerator = **GPU T4 x2** | Internet = **On** | keep the notebook **Private** |
Add-ons -> Secrets: tick `HF_TOKEN` and `OPENAI_API_KEY` (or `GEMINI_API_KEY`) for THIS notebook.

**Two-step routine for every model**
1. Leave `PILOT_ONLY = True`, run all cells (2-5 minutes). It checks your setup and the extraction probe, and prints either
   `PILOT PASSED` or exactly what is wrong. Nothing long-running starts.
2. If it passed: set `PILOT_ONLY = False` and use *Save Version -> Save & Run All*. It then runs generation, judging, analysis and packaging by itself.

The notebook never crashes with a traceback on a setup problem: it prints a boxed message and stops. Everything finished is kept.
Order of models: `gpt4o_mini` -> `gemini35_flash_lite` -> `llama31_8b` -> `mistral7b_v03`, then `02_compare_models`."""),
    code("""# ===================== CONFIG - the only cell you normally edit =====================
TARGET = "gpt4o_mini"        # gpt4o_mini | gemini35_flash_lite | llama31_8b | mistral7b_v03
PILOT_ONLY = True            # True = quick check only. Set False for the full run once the pilot has passed.
EXTRACT_VARIANT = "auto"     # FIRST model: "auto" (finds a wording that works). Other models: paste the wording it chose, e.g. "v3".

RUN_NAME = "final_run_v1"
MAX_MINUTES = 600            # generation stops cleanly after this long (Kaggle cuts a run at 12 h; judging + analysis need ~30 min after). Re-run with RESUME_FROM to continue
RESUME_FROM = None           # "auto" = use an earlier version's output that you added via Add Input -> Notebook Output Files
JUDGE_ONLY = False           # True = the generation already finished earlier; only judge + analyse + package (use with RESUME_FROM = "auto")

OPENAI_MAX_RPM = 40          # raise if your OpenAI tier allows
GEMINI_MODEL_ID = "gemini-3.5-flash-lite"
GEMINI_MAX_RPM = 12          # the free tier is slow (your old pipeline used 1 call per 4.5 s = 13/min); lower if you see many 429s
GEMINI_REASONING_EFFORT = None   # if Gemini replies come back EMPTY, try "low" or "none"
JUDGE_PRECISION = "fp16"     # "fp16" (preferred) or "4bit" if memory is a problem
"""),
    code("%%writefile overload_lib.py\n" + lib),
    code("""!pip -q install rapidfuzz tiktoken"""),
    code("""import os, sys
sys.path.insert(0, os.getcwd())
import overload_lib as ol
STATUS = ol.run_all(TARGET, RUN_NAME, extract_variant=EXTRACT_VARIANT, max_minutes=MAX_MINUTES,
                    resume_from=RESUME_FROM, pilot_only=PILOT_ONLY, openai_max_rpm=OPENAI_MAX_RPM,
                    gemini_model_id=GEMINI_MODEL_ID, gemini_max_rpm=GEMINI_MAX_RPM,
                    gemini_reasoning_effort=GEMINI_REASONING_EFFORT, judge_precision=JUDGE_PRECISION,
                    judge_only=JUDGE_ONLY)
print("\\nOK:", STATUS.get("ok"))"""),
    code("""from pathlib import Path
f = Path(STATUS.get("folder", ".")) / "06_figures" / "fig_heatmaps_harmful.png"
if f.exists():
    from IPython.display import Image, display
    display(Image(str(f)))"""),
]
save(run, "01_run_model.ipynb")

# ------------------------------------------------------------------ 02_compare_models
cmp_ = [
    md("""# Cross-model tables and figures (CPU only - no GPU needed)

1. Put the four model folders (`gpt4o_mini`, `gemini35_flash_lite`, `llama31_8b`, `mistral7b_v03`) under one folder, e.g. unzip the four
   `*_SHAREABLE.zip` files into `honours_results/final_run_v1/`. On Kaggle: upload them as a **private Dataset**.
2. Set `ROOT` below. 3. Run all cells. Output goes to `ROOT/cross_model/` plus `00_REPORT_MAP.md`.
4. To include Cohen's kappa, place each model's `audit_key.csv` and `audit_labels.csv` in that model's `05_analysis_tables/`."""),
    code('''ROOT = "/kaggle/input/honours-results/honours_results/final_run_v1"   # <- change me (or a local path)
WORK = "/kaggle/working/final_run_v1"                                  # writable copy (Kaggle inputs are read-only)'''),
    code("%%writefile overload_lib.py\n" + lib),
    code('''!pip -q install rapidfuzz
import shutil, os, sys
sys.path.insert(0, os.getcwd())
import overload_lib as ol
if not os.path.exists(WORK):
    shutil.copytree(ROOT, WORK)
ol.finalize(WORK)
print(open(WORK + "/00_REPORT_MAP.md").read())'''),
]
save(cmp_, "02_compare_models.ipynb")
