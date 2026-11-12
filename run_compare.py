"""Cross-model step on any machine (CPU only).
    python run_compare.py path/to/honours_results/final_run_v1
Needs, for each model folder: 05_analysis_tables/, 03_pilot/, (07_mechanistic/ for Llama).
If audit_key.csv + audit_labels.csv are present, Cohen's kappa is computed too."""
import sys
import overload_lib as ol
ol.finalize(sys.argv[1])
