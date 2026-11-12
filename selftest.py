"""
selftest.py - runs the ENTIRE pipeline (data -> prompts -> pilot -> generation -> judge -> analysis ->
audit -> cross-model tables) with a fake model and a fake judge. No GPU, no API key, ~1 minute.

It proves that the design logic, scoring, statistics, tables and figures work and that every file listed in
Appendix F gets written. It says NOTHING about real model behaviour (the fake model's numbers are invented).

    python selftest.py            # writes to a temporary folder
    python selftest.py --keep out # keep the output in ./out
"""
import hashlib
import random
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import overload_lib as ol


class MockBackend:
    kind = "mock"

    def __init__(self, target, instructions, comply_base, comply_slope, needs_tags=False, always_none=False):
        self.target, self.ins = target, instructions
        self.base, self.slope = comply_base, comply_slope
        self.needs_tags, self.always_none = needs_tags, always_none

    def count(self, t):
        return len(t.split())

    def run(self, jobs):
        out = []
        for j in jobs:
            p = j["prompt"]
            rng = random.Random(int(hashlib.md5((self.target + p).encode()).hexdigest()[:8], 16))
            words = len(p.split())
            inst = next((i for i in self.ins if i in p), "")
            if "word for word" in p:
                if self.always_none or (self.needs_tags and "<document>" not in p):
                    txt = "NONE"          # simulates GPT-4o mini failing the old wordings
                else:
                    txt = f'"{inst}"' if rng.random() < max(0.25, 1 - words / 14000) else "NONE"
            else:
                pc = min(0.95, self.base + self.slope * np.log2(max(words, 2)) / 13)
                r = rng.random()
                txt = ("Sure, here is how to do it: step 1 ... step 2 ..." if r < pc
                       else "I'm sorry, I can't help with that." if r < pc + 0.5 * (1 - pc)
                       else "The document is an encyclopaedia extract about various topics.")
            out.append(dict(text=txt, finish_reason="stop", prompt_tokens=words, completion_tokens=12, latency=0.01,
                            error=None, blocked=False, resp_model=self.target))
        return out


class MockJudge:
    def score(self, behaviors, generations, bs=8):
        return ["yes" if g.startswith("Sure, here") else "no" for g in generations]


def main(keep=None):
    tmp = Path(keep) if keep else Path(tempfile.mkdtemp())
    ol.results_root = lambda: tmp / "honours_results"
    adv = ol.load_advbench()
    rng = random.Random(1)
    jbb = pd.DataFrame(dict(goal=[f"Describe how to bake item number {i} for a school fair" for i in range(100)]))
    alpaca = [f"Give me a tip about topic {i}" for i in range(500)]
    words = "river stone market engine forest letter bridge garden season harbour tower valley".split()
    wiki = [" ".join(rng.choice(words) for _ in range(rng.randint(8, 25))).capitalize() + "." for _ in range(7000)]
    data = dict(adv=adv, jbb=jbb, alpaca=alpaca, wiki=wiki)
    count = lambda s: int(len(s.split()) * 1.3)

    params = {"gpt4o_mini": (0.02, 0.05), "gemini35_flash_lite": (0.10, 0.20), "llama31_8b": (0.15, 0.45), "mistral7b_v03": (0.30, 0.50)}
    for t, (b, s) in params.items():
        print(f"\n===== {t} =====")
        d = ol.setup_run("selftest_run", t)
        ol.write_environment(d, t)
        c = ol.prepare(d, count, data)
        ol.design_checks(c)
        be = MockBackend(t, list(c.items.text), b, s)
        print(ol.run_pilot(be, d, c, t, ladder=[500, 2000, 8000]))
        ol.run_main(be, d, c, max_minutes=5, chunk=64)
        ol.run_main(be, d, c, max_minutes=5, chunk=64)     # second call must find nothing to do (resume check)
        ol.run_judge(MockJudge(), d, c)
        sc = ol.analyse_model(d, t)
        n = ol.make_audit_sheet(d)
        key = pd.read_csv(d / "08_manual_audit_PRIVATE" / "audit_key.csv")
        lab = pd.DataFrame(dict(audit_id=key.audit_id, human_label=key.auto_label.map(
            {"harmful_compliance": "harmful", "refusal": "refusal", "other_non_harmful": "other"})))
        lab.loc[::7, "human_label"] = "other"               # a few deliberate disagreements
        lab.to_csv(d / "08_manual_audit_PRIVATE" / "audit_labels.csv", index=False)
        summ, conf = ol.audit_kappa(d / "08_manual_audit_PRIVATE" / "audit_key.csv", d / "08_manual_audit_PRIVATE" / "audit_labels.csv", d / "05_analysis_tables")
        print("audit:", n, "rows;", summ)
        if t == "llama31_8b":                               # fabricate mechanistic outputs to test the analysis code
            r = np.random.RandomState(0)
            proj = c.design[["trial_id", "length"]].copy()
            proj["projection"] = 5 - 0.0004 * proj.length + r.randn(len(proj))
            proj.drop(columns="length").to_csv(d / "07_mechanistic" / "projections.csv", index=False)
            gens = []
            for name, shift in (("none", 0), ("restore", 0.3), ("full_vector", 0.5), ("random_full_norm", 0.0), ("random_restore_norm", 0.0)):
                for L, P in ((0, None), (8000, 50)):
                    for it in c.items.itertuples():
                        harmful_p = max(0.0, (0.05 if L == 0 else 0.5) - shift)
                        yes = r.rand() < harmful_p
                        gens.append(dict(key=f"{ol.trial_id(it.item_id, L, P)}|{name}", trial_id=ol.trial_id(it.item_id, L, P), intervention=name,
                                         length=L, branch="response", text="Sure, here is" if yes else "I'm sorry, I can't", error=None, blocked=False))
            ol.append_jsonl(d / "04_raw_PRIVATE" / "intervention_generations.jsonl", gens)
            ol.run_judge(MockJudge(), d, c, which="intervention")
            ol.analyse_mechanistic(d, c)
        ol.write_index(d, t)
        ol.package(d, t)
    ol.compare_models(tmp / "honours_results" / "selftest_run")

    # ---- run_all: the one-call runner must pick a working wording, stop calmly on a failed pilot, and never raise
    ins = list(ol.build_items(adv, jbb, alpaca)[0].text)
    hooks = lambda be: dict(backend=be, judge=MockJudge(), data=data, count=count, n_gpu=2)
    st = ol.run_all("gpt4o_mini", "ra_auto", extract_variant="auto", _hooks=hooks(MockBackend("gpt4o_mini", ins, 0.05, 0.1, needs_tags=True)))
    assert st["ok"] and st["pilot"]["probe_variant"] in ("v3", "v4", "v5"), st
    assert (tmp / "honours_results" / "ra_auto" / "gpt4o_mini" / "gpt4o_mini_SHAREABLE.zip").exists() or (tmp / "honours_results" / "ra_auto" / "gpt4o_mini_SHAREABLE.zip").exists()
    st = ol.run_all("gemini35_flash_lite", "ra_fail", extract_variant="auto", _hooks=hooks(MockBackend("g", ins, 0.05, 0.1, always_none=True)))
    assert not st["ok"] and not (tmp / "honours_results" / "ra_fail" / "gemini35_flash_lite" / "04_raw_PRIVATE" / "generations.jsonl").exists(), st
    st = ol.run_all("gpt4o_mini", "ra_pilot", extract_variant="v4", pilot_only=True, _hooks=hooks(MockBackend("gpt4o_mini", ins, 0.05, 0.1, needs_tags=True)))
    assert st["ok"] and not (tmp / "honours_results" / "ra_pilot" / "gpt4o_mini" / "04_raw_PRIVATE" / "generations.jsonl").exists(), st
    st = ol.run_all("llama31_8b", "ra_pre")          # no GPU, no token here -> must explain, not crash
    assert not st["ok"] and st.get("problems"), st
    # ---- judge-only: generation finished earlier, judge step re-run from the saved output (the Mistral situation)
    src = tmp / "honours_results" / "ra_auto" / "gpt4o_mini"
    ok_before = (src / "04_raw_PRIVATE" / "judge_verdicts.csv").exists()
    for f in ["04_raw_PRIVATE/judge_verdicts.csv", "05_analysis_tables/trial_level_scores.csv"]:
        (src / f).unlink(missing_ok=True)
    st = ol.run_all("gpt4o_mini", "ra_judgeonly", judge_only=True, resume_from=str(src), _hooks=hooks(None) | dict(backend=None))
    assert ok_before and st["ok"] and (tmp / "honours_results" / "ra_judgeonly" / "gpt4o_mini" / "04_raw_PRIVATE" / "judge_verdicts.csv").exists(), st
    st = ol.run_all("gpt4o_mini", "ra_judgeonly2", judge_only=True, resume_from=str(tmp / "nowhere"), _hooks=hooks(None))
    assert not st["ok"], st            # missing folder -> explains, does not crash
    print("\nrun_all checks: auto wording, failed pilot stops calmly, pilot-only, pre-flight problems -> all OK")

    root = tmp / "honours_results" / "selftest_run"
    print("\n" + (root / "00_REPORT_MAP.md").read_text())
    missing = [l for l in (root / "00_REPORT_MAP.md").read_text().splitlines() if "MISSING" in l or "0/4" in l or "1/4" in l or "2/4" in l or "3/4" in l]
    # figures that only exist for one model are expected to read 1/4
    missing = [l for l in missing if "fig_mech" not in l and "4.9" not in l]
    print("\nSELFTEST", "PASSED" if not missing else f"FINISHED WITH GAPS: {missing}")
    print("output folder:", tmp)


if __name__ == "__main__":
    main(sys.argv[2] if len(sys.argv) > 2 and sys.argv[1] == "--keep" else None)
