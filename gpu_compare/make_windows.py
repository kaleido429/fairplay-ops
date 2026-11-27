"""프레임별 에임 CSV -> preprocessed_data.xlsx와 같은 형식의 윈도우 특징 엑셀.

기존 엑셀에 있는 영상만 남기고(같은 255개), 열 순서도 기존 엑셀과 맞춤.
사용: python make_windows.py <reeval 폴더> <프레임 csv> <출력 xlsx>
"""
import io
import sys

import numpy as np
import pandas as pd

reeval, src, out = sys.argv[1:4]
sys.path.insert(0, reeval)
from holdout_eval import window_features  # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
orig = pd.read_excel(f"{reeval}/preprocessed_data.xlsx")
fr = pd.read_csv(src)
print("추출된 영상", fr.filename.nunique(), "| 기존 엑셀 영상", orig.filename.nunique())
print("기존 엑셀에 없는 영상:", sorted(set(fr.filename) - set(orig.filename)))
print("추출 안 된 영상:", sorted(set(orig.filename) - set(fr.filename)))
w = window_features(fr[fr.filename.isin(orig.filename.unique())])
w["cheat"] = w.filename.str.startswith("cheats/").astype(int)
w = w[list(orig.columns)]
w.to_excel(out, index=False)
print("윈도우", len(w), "| 기존", len(orig))

m = orig.merge(w, on=["filename", "start_frame"], suffixes=("_o", "_n"))
print("짝이 맞는 윈도우", len(m))
cols = [c for c in orig.columns if c not in ("filename", "cheat", "start_frame")]
rel = pd.Series({c: np.nanmean(np.abs(m[c + "_o"] - m[c + "_n"])) / (orig[c].std() + 1e-12) for c in cols})
print("특징 차이(기존 표준편차 대비) 중앙값 %.4f, 상위 5개:" % rel.median())
print(rel.sort_values(ascending=False).head(5).round(4).to_string())
