"""Rehearse the evaluation server before submitting.

Two failure modes are worth real money here and neither raises an exception on
its own:

* preprocessing drift between training and submit/script.py -- silently wrong
  predictions. Generated code makes this structurally impossible, so we just
  assert it.
* an artifact that will not unpickle on the server. The server runs numpy
  1.26.4; a pickle written under numpy 2.x dies with a `numpy._core` ImportError
  at load time. Run this with .venv-submit/Scripts/python.exe.

The script then builds a fake ./data + ./model tree, runs script.py exactly as
the server would, and checks the output shape, range and wall clock.

    ../.venv-submit/Scripts/python.exe verify_v4.py
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time

import joblib
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

import fe  # noqa: E402
import nnpred  # noqa: E402

N_FAKE = 250000
TIME_LIMIT = 600.0


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    print(f"numpy {np.__version__} / pandas {pd.__version__}")
    art_path = os.path.join(ROOT, "submit", "model", "model.joblib")
    t = time.time()
    art = joblib.load(art_path)
    print(f"[OK] artifact loads ({os.path.getsize(art_path) / 1e6:.1f} MB, "
          f"{time.time() - t:.1f}s), "
          + " + ".join(
              f"{g['tag']} x{len(g['nn']['nets']) if g['kind'] == 'nn' else len(g['models'])}"
              f" w={g['weight']:.3f}" for g in art["groups"])
          + f", {len(art['columns'])} features")

    script_path = os.path.join(ROOT, "submit", "script.py")
    submit = load_module(script_path, "submit_script")

    df = pd.read_csv(os.path.join(ROOT, "open", "data", "train.csv"),
                     encoding="utf-8-sig", nrows=20000).drop(columns=["control_success"])
    a = fe.build_features(df, art["ctx"], art["columns"])
    b = submit.build_features(df, art["ctx"], art["columns"])
    av, bv = a.to_numpy(dtype="float64"), b.to_numpy(dtype="float64")
    if not np.isclose(av, bv, equal_nan=True).all():
        raise SystemExit("[FAIL] script.py preprocessing differs from fe.py "
                         "-- rerun build_submit.py")
    print(f"[OK] script.py matches fe.py on {len(df)} rows x {len(art['columns'])} features")

    nn = next((g for g in art["groups"] if g["kind"] == "nn"), None)
    if nn is not None:
        pa = nnpred.nn_predict(a, nn["nn"])
        pb = submit.nn_predict(a, nn["nn"])
        if not np.allclose(pa, pb, atol=1e-9):
            raise SystemExit("[FAIL] script.py net inference differs from nnpred.py")
        print(f"[OK] net inference matches ({len(nn['nn']['nets'])} nets, "
              f"mean={pa.mean():.4f}) -- numpy only, no torch on the server")

    # --- full server rehearsal on a 2025-shaped test set -----------------------
    big = pd.read_csv(os.path.join(ROOT, "open", "data", "train.csv"),
                      encoding="utf-8-sig", skiprows=range(1, 1475092 - N_FAKE + 1))
    big = big.drop(columns=["control_success"])
    big["season"] = int(max(art["trained_seasons"])) + 1     # unseen season, as in the real run
    big[fe.ID_COL] = [f"TEST_{i:06d}" for i in range(len(big))]
    sample = pd.DataFrame({fe.ID_COL: big[fe.ID_COL], fe.TARGET_COL: 0.5})

    tmp = tempfile.mkdtemp(prefix="server_rehearsal_")
    os.makedirs(os.path.join(tmp, "data"))
    os.makedirs(os.path.join(tmp, "model"))
    big.to_csv(os.path.join(tmp, "data", "test.csv"), index=False, encoding="utf-8")
    sample.to_csv(os.path.join(tmp, "data", "sample_submission.csv"), index=False,
                  encoding="utf-8")
    shutil.copy2(art_path, os.path.join(tmp, "model", "model.joblib"))
    shutil.copy2(script_path, os.path.join(tmp, "script.py"))

    t = time.time()
    r = subprocess.run([sys.executable, "script.py"], cwd=tmp,
                       capture_output=True, text=True)
    elapsed = time.time() - t
    print("--- script.py stdout ---")
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr[-3000:])
        raise SystemExit("[FAIL] script.py crashed")

    out = pd.read_csv(os.path.join(tmp, "output", "submission.csv"))
    assert list(out.columns) == [fe.ID_COL, fe.TARGET_COL], out.columns.tolist()
    assert len(out) == len(sample), (len(out), len(sample))
    assert out[fe.ID_COL].tolist() == sample[fe.ID_COL].tolist(), "row_id order changed"
    p = out[fe.TARGET_COL].to_numpy()
    assert np.isfinite(p).all() and (p > 0).all() and (p < 1).all(), "bad probabilities"
    print(f"[OK] {len(out)} rows, mean={p.mean():.4f}, "
          f"range [{p.min():.4f}, {p.max():.4f}]")
    print(f"[OK] wall clock {elapsed:.1f}s for {len(out)} rows "
          f"(limit {TIME_LIMIT:.0f}s)")
    if elapsed > TIME_LIMIT * 0.5:
        print("[WARN] over half the time budget")
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\napplied target {art['target_rate']:.4f} "
          f"(forecast {art['forecast']:.4f} at {art['offset_scale']:.2f} strength, "
          f"reference {art['ref_mean']:.4f})")


if __name__ == "__main__":
    main()
