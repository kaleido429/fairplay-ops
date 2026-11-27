"""전체 영상(학습 + 홀드아웃)을 합쳐 중첩 교차검증.

바깥 5묶음: 4묶음으로 학습하고 1묶음을 채점. 모든 영상이 '자기를 보지 않은 모델'에게 한 번씩 채점된다.
안쪽 5묶음: 바깥 학습 4묶음 안에서만 OOF 점수를 구해 임계값(정상 영상 최고 점수)을 정한다.
            분할 1회(single)와 분할 INNER_SEEDS회 평균(avg) 두 방식.
바깥 분할 시드를 OUTER_SEEDS번 바꿔 반복해 분할에 따른 흔들림도 본다.

사용: python full_cv.py [fairplay.db] [출력 접두어]
"""
import io
import json
import sqlite3
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

import features

OUTER_SEEDS, INNER_SEEDS = 5, 10


def video_scores(g, p):
    return pd.DataFrame({"g": g, "p": p}).groupby("g")["p"].agg(["max", "mean"])


def thresholds(X, y, g):
    """학습 묶음 안에서만 임계값 결정. {(rule, method): value}"""
    normal = np.unique(g[y == 0])
    per_seed = []
    for s in range(INNER_SEEDS):
        oof = np.zeros(len(y))
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(X, y, g):
            oof[b] = features.fit(X[a], y[a]).predict_proba(X[b])[:, 1]
        per_seed.append(video_scores(g, oof).loc[normal])
    avg = sum(per_seed) / INNER_SEEDS
    out = {}
    for rule in ("max", "mean"):
        out[rule, "single"] = float(per_seed[0][rule].max())
        out[rule, "avg"] = float(avg[rule].max())
    return out


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    db = sys.argv[1] if len(sys.argv) > 1 else "fairplay.db"
    prefix = sys.argv[2] if len(sys.argv) > 2 else "full_cv"
    con = sqlite3.connect(db)
    vids = pd.read_sql("SELECT video_id, filename, split, is_cheat FROM videos WHERE status = 'extracted'", con)
    fr = pd.read_sql("SELECT video_id, dx, dy FROM frames ORDER BY video_id, frame", con)
    t0 = time.perf_counter()
    with Pool(6) as pool:
        w = pd.concat(pool.map(features.video_windows,
                               [(int(v), d[["dx", "dy"]].to_numpy()) for v, d in fr.groupby("video_id")], chunksize=4),
                      ignore_index=True)
    w = w.merge(vids, on="video_id")
    cols = [c for c in w.columns if c not in ("video_id", "start_frame", "filename", "split", "is_cheat")]
    X, y, g = w[cols].to_numpy(float), w.is_cheat.to_numpy(), w.video_id.to_numpy()
    vinfo = vids.set_index("video_id").loc[np.unique(g)]
    print(f"영상 {len(vinfo)}개 (치트 {int(vinfo.is_cheat.sum())}, 정상 {int((vinfo.is_cheat == 0).sum())}), "
          f"윈도우 {len(w)}개, 특징 {len(cols)}개, {time.perf_counter() - t0:.0f}초", flush=True)

    rows, thr_rows, aucs = [], [], []
    for os_ in range(OUTER_SEEDS):
        oof = np.zeros(len(y))
        for k, (a, b) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=os_).split(X, y, g)):
            thr = thresholds(X[a], y[a], g[a])
            oof[b] = features.fit(X[a], y[a]).predict_proba(X[b])[:, 1]
            sc = video_scores(g[b], oof[b])
            for (rule, method), t in thr.items():
                thr_rows.append(dict(outer_seed=os_, fold=k, rule=rule, method=method, threshold=t))
                for vid, s in sc[rule].items():
                    rows.append(dict(outer_seed=os_, fold=k, video_id=int(vid), rule=rule, method=method,
                                     score=float(s), threshold=t, flagged=int(s > t)))
            print(f"  시드 {os_} 묶음 {k}: 평가 영상 {len(sc)}개, {time.perf_counter() - t0:.0f}초", flush=True)
        sc = video_scores(g, oof).join(vinfo.is_cheat)
        aucs.append(dict(outer_seed=os_, window_auc=roc_auc_score(y, oof),
                         video_auc_max=roc_auc_score(sc.is_cheat, sc["max"]),
                         video_auc_mean=roc_auc_score(sc.is_cheat, sc["mean"])))

    r = pd.DataFrame(rows).merge(vinfo.reset_index(), on="video_id")
    r.to_csv(prefix + "_videos.csv", index=False)
    pd.DataFrame(thr_rows).to_csv(prefix + "_thresholds.csv", index=False)

    def agg(d):
        c, n = d[d.is_cheat == 1], d[d.is_cheat == 0]
        return pd.Series(dict(cheats=len(c), caught=int(c.flagged.sum()), recall=c.flagged.mean(),
                              normals=len(n), fp=int(n.flagged.sum()), fpr=n.flagged.mean()))

    per_seed = r.groupby(["rule", "method", "outer_seed"]).apply(agg, include_groups=False).reset_index()
    summary = per_seed.groupby(["rule", "method"]).agg(
        caught_mean=("caught", "mean"), caught_min=("caught", "min"), caught_max=("caught", "max"),
        recall_mean=("recall", "mean"), fp_mean=("fp", "mean"), fp_min=("fp", "min"), fp_max=("fp", "max"),
        fpr_mean=("fpr", "mean")).reset_index()
    by_origin = r.groupby(["rule", "method", "split", "outer_seed"]).apply(agg, include_groups=False).reset_index() \
        .groupby(["rule", "method", "split"])[["cheats", "caught", "recall", "normals", "fp", "fpr"]].mean().reset_index()
    thr = pd.DataFrame(thr_rows).groupby(["rule", "method"]).threshold.agg(["min", "median", "max"]).reset_index()
    auc = pd.DataFrame(aucs)

    pd.set_option("display.width", 200)
    print("\n== 바깥 시드별\n", per_seed.round(4).to_string(index=False))
    print("\n== 요약 (바깥 시드 5개)\n", summary.round(4).to_string(index=False))
    print("\n== 영상 출처별 평균 (train = 기존 학습 영상, holdout = 기존 홀드아웃 영상)\n", by_origin.round(4).to_string(index=False))
    print("\n== 임계값 범위 (25개 묶음)\n", thr.round(5).to_string(index=False))
    print("\n== AUC\n", auc.round(4).to_string(index=False))
    json.dump(dict(videos=len(vinfo), cheats=int(vinfo.is_cheat.sum()), windows=len(w),
                   outer_seeds=OUTER_SEEDS, inner_seeds=INNER_SEEDS,
                   per_seed=per_seed.to_dict("records"), summary=summary.to_dict("records"),
                   by_origin=by_origin.to_dict("records"), thresholds=thr.to_dict("records"),
                   auc=auc.to_dict("records")),
              open(prefix + "_summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n전체 {time.perf_counter() - t0:.0f}초")
