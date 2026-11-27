"""임계값이 분할 시드에 얼마나 좌우되는지 확인.

holdout_eval.py는 임계값을 5-fold OOF 1회(시드 0)의 정상 영상 최고 점수로 정한다.
여기서는 시드만 0~19로 바꿔 임계값과 홀드아웃 결과의 변동 폭을 보고,
시드 20개의 영상 점수를 평균한 뒤 최고 점수를 쓰는 방식(평균 임계값)과 비교한다.
모델(기존 255개 전체 학습)과 홀드아웃 점수는 모든 경우에 같다.

사용: python threshold_check.py preprocessed_data.xlsx holdout_aim.csv threshold_check.json
"""
import io
import json
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from holdout_eval import fit, video_scores, window_features

N_SEEDS = 20

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    orig = pd.read_excel(sys.argv[1])
    cols = [c for c in orig.columns if c not in ("filename", "cheat", "start_frame")]
    X, y, g = orig[cols].to_numpy(float), orig["cheat"].to_numpy(), orig["filename"].to_numpy()
    label = orig.groupby("filename")["cheat"].first()
    normal = label.index[label == 0]

    hold = window_features(pd.read_csv(sys.argv[2]))
    hv = video_scores(hold["filename"].to_numpy(), fit(X, y).predict_proba(hold[cols].to_numpy(float))[:, 1])
    is_cheat = hv.index.str.startswith("cheats/")

    # 시드별 OOF 영상 점수 (정상 영상만)
    scores = []
    for seed in range(N_SEEDS):
        oof = np.zeros(len(orig))
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
            oof[va] = fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
        scores.append(video_scores(g, oof).loc[normal])
        print(f"seed {seed} done", flush=True)
    avg = sum(scores) / N_SEEDS

    def apply(col, thr):
        flag = (hv[col] > thr).to_numpy()
        return int((flag & is_cheat).sum()), int((flag & ~is_cheat).sum())

    out = {"n_cheat": int(is_cheat.sum()), "n_normal": int((~is_cheat).sum()), "n_seeds": N_SEEDS}
    print(f"\n홀드아웃: 치트 {out['n_cheat']}개, 정상 {out['n_normal']}개 / 시드 {N_SEEDS}개")
    for col, name in [("pmax", "기존 설계 (윈도우 최댓값)"), ("pmean", "개선안 (영상 평균)")]:
        thr = [float(s[col].max()) for s in scores]
        res = [apply(col, t) for t in thr]
        tp, fp = [r[0] for r in res], [r[1] for r in res]
        t_avg = float(avg[col].max())
        tp_avg, fp_avg = apply(col, t_avg)
        out[col] = dict(thr_by_seed=thr, tp_by_seed=tp, fp_by_seed=fp, tp_median=float(np.median(tp)),
                        thr_avg=t_avg, tp_avg=tp_avg, fp_avg=fp_avg)
        print(f"[{name}]")
        print(f"  시드별: 임계값 {min(thr):.5f}~{max(thr):.5f} | 적발 {min(tp)}~{max(tp)} (중앙값 {np.median(tp):g}) | "
              f"오탐 {min(fp)}~{max(fp)} | 시드 0: 적발 {tp[0]}, 오탐 {fp[0]}")
        print(f"  평균 임계값 {t_avg:.5f}: 적발 {tp_avg}, 오탐 {fp_avg}")
    json.dump(out, open(sys.argv[3], "w"), indent=2)
