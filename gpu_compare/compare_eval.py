"""compare_flow.py 결과로 CPU/GPU 특징 차이와 판정 변화 확인.

사용: python compare_eval.py <reeval 폴더> <compare csv>
(일반 venv로 실행. reeval 폴더의 extract_aim.per_frame, holdout_eval 함수 재사용)
"""
import io
import sys

import numpy as np
import pandas as pd

reeval, comp = sys.argv[1], sys.argv[2]
sys.path.insert(0, reeval)
from extract_aim import per_frame  # noqa: E402
from holdout_eval import fit, video_scores, window_features  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
pd.set_option("display.width", 220)
c = pd.read_csv(comp)
names = sorted(c.filename.unique())
print(f"영상 {len(names)}개, 프레임 {len(c)}개")

# 0) 이 빌드의 CPU 결과가 기존(pip 빌드) 결과와 같은지
old = pd.read_csv(f"{reeval}/holdout_aim.csv")
m = c.merge(old[["filename", "frame", "dx", "dy"]], on=["filename", "frame"])
print(f"[기준 확인] 새 빌드 CPU vs 기존 CSV: 프레임 {len(m)}개, 최대 차이 "
      f"{max(np.abs(m.dx_cpu - m.dx).max(), np.abs(m.dy_cpu - m.dy).max()):.2e}")

# 1) 프레임 단위 차이
d = np.hypot(c.dx_cpu - c.dx_gpu, c.dy_cpu - c.dy_gpu)
mag = np.hypot(c.dx_cpu, c.dy_cpu)
print("\n[프레임별 평균 흐름 차이 |cpu-gpu| (px/frame)]")
print(f"  중앙값 {d.median():.2e} | 평균 {d.mean():.2e} | 99% {d.quantile(.99):.2e} | 최대 {d.max():.2e}")
print(f"  참고: 흐름 크기(속도) 중앙값 {mag.median():.3f}, 상대 차이 중앙값 {(d / mag.clip(lower=1e-9)).median():.2e}")
print(f"  픽셀 단위 차이: 평균 {c.px_absdiff_mean.mean():.2e}, 최대 {c.px_absdiff_max.max():.2f}")

# 2) 프레임 특징과 윈도우 특징
def frames_of(tag):
    return pd.concat(per_frame(n, g[[f"dx_{tag}", f"dy_{tag}"]].to_numpy()) for n, g in c.groupby("filename", sort=True))


fc, fg = frames_of("cpu"), frames_of("gpu")
print("\n[프레임 특징별 평균 절대 차이 / CPU 값의 표준편차]")
for col in ["dx", "dy", "velocity", "acceleration", "jerk", "angle_deg", "angle_change"]:
    a, b = fc[col].to_numpy(), fg[col].to_numpy()
    diff = np.abs(a - b)
    if col == "angle_deg":
        diff = np.minimum(diff, 360 - diff)
    print(f"  {col:13s} {np.nanmean(diff):.3e} / {np.nanstd(a):.3f}")

wc, wg = window_features(fc), window_features(fg)

# 3) 모델 점수와 판정 (holdout_eval과 같은 절차)
orig = pd.read_excel(f"{reeval}/preprocessed_data.xlsx")
cols = [k for k in orig.columns if k not in ("filename", "cheat", "start_frame")]
X, y, g = orig[cols].to_numpy(float), orig["cheat"].to_numpy(), orig["filename"].to_numpy()
oof = np.zeros(len(orig))
for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, y, g):
    oof[va] = fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
vs = video_scores(g, oof).join(orig.groupby("filename")["cheat"].first())
thr_max, thr_mean = vs.loc[vs.cheat == 0, "pmax"].max(), vs.loc[vs.cheat == 0, "pmean"].max()
model = fit(X, y)

rel = (wc[cols] - wg[cols]).abs().to_numpy(float) / (orig[cols].std().to_numpy(float) + 1e-12)
worst = pd.Series(np.nanmean(rel, axis=0), index=cols).sort_values(ascending=False)
print("\n[윈도우 특징 차이가 큰 상위 8개 (학습 데이터 표준편차 대비)]")
print(worst.head(8).round(4).to_string())

pc = model.predict_proba(wc[cols].to_numpy(float))[:, 1]
pg = model.predict_proba(wg[cols].to_numpy(float))[:, 1]
print(f"\n[윈도우 확률 차이] 평균 {np.abs(pc - pg).mean():.2e} | 최대 {np.abs(pc - pg).max():.3f} | 윈도우 {len(pc)}개")
sc, sg = video_scores(wc.filename.to_numpy(), pc), video_scores(wg.filename.to_numpy(), pg)
r = sc.join(sg, lsuffix="_cpu", rsuffix="_gpu")
r["cheat"] = r.index.str.startswith("cheats/").astype(int)
for k, col, thr in [("기존 설계(max)", "pmax", thr_max), ("개선안(mean)", "pmean", thr_mean)]:
    a, b = r[f"{col}_cpu"] > thr, r[f"{col}_gpu"] > thr
    pos, neg = r.cheat == 1, r.cheat == 0
    print(f"[{k}] CPU: 적발 {int(a[pos].sum())}/{int(pos.sum())}, 오탐 {int(a[neg].sum())}/{int(neg.sum())} | "
          f"GPU: 적발 {int(b[pos].sum())}/{int(pos.sum())}, 오탐 {int(b[neg].sum())}/{int(neg.sum())} | "
          f"판정이 바뀐 영상 {int((a != b).sum())}개")
    if (a != b).any():
        print(r.loc[a != b, [f"{col}_cpu", f"{col}_gpu", "cheat"]].round(5).to_string())
print(f"영상 점수 차이: pmean 최대 {np.abs(r.pmean_cpu - r.pmean_gpu).max():.4f}, pmax 최대 {np.abs(r.pmax_cpu - r.pmax_gpu).max():.4f}")

# 4) 속도
t = pd.read_csv(comp.replace(".csv", "_time.csv"))
print(f"\n[시간 합계, 초] 디코딩 {t.t_decode.sum():.0f} | CPU flow {t.t_cpu.sum():.0f} | GPU flow {t.t_gpu.sum():.0f} "
      f"| flow만 {t.t_cpu.sum() / t.t_gpu.sum():.1f}배, 디코딩 포함 전체 {(t.t_decode.sum() + t.t_cpu.sum()) / (t.t_decode.sum() + t.t_gpu.sum()):.1f}배")
t["res"] = t.width.astype(str) + "x" + t.height.astype(str)
print((t.groupby("res")[["t_decode", "t_cpu", "t_gpu"]].sum().div(t.groupby("res").frames.sum(), axis=0) * 1000).round(1)
      .rename(columns=lambda s: s + " ms/frame").to_string())
r.to_csv(comp.replace(".csv", "_videos.csv"))
