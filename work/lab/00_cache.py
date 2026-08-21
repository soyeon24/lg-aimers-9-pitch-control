"""train.csv -> parquet 캐시. 이후 실험 로딩을 초 단위로 줄인다."""
import os, time
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "open", "data")
OUT = os.path.join(HERE, "cache")
os.makedirs(OUT, exist_ok=True)

t = time.time()
df = pd.read_csv(os.path.join(DATA, "train.csv"), encoding="utf-8-sig")
print("train", df.shape, f"{time.time()-t:.0f}s")
print(df.dtypes.to_string())
df.to_parquet(os.path.join(OUT, "train.parquet"), index=False)
print("saved", f"{time.time()-t:.0f}s")
