"""임계값 결정 방법 비교. 방법 선택은 학습 데이터(255개 영상)만으로 한다.

비교하는 방법 (정상 영상의 OOF 점수로 임계값을 정하는 방식)
- max    : 정상 영상 점수의 최댓값 (기존 방식, 분할 1회)
- q99    : 정상 영상 점수의 99% 분위수 (분할 1회)
- avgmax : 분할을 여러 번 바꿔 얻은 영상 점수를 평균한 뒤 최댓값
- avgq99 : 위 평균 점수의 99% 분위수

1부: 학습 데이터 전체에서 분할 시드만 바꿨을 때 임계값이 얼마나 흔들리는가
2부: 중첩 교차검증(바깥 5-fold x 10회)으로 방법별 재현율·오탐률
3부: 홀드아웃 적용 (참고용. 홀드아웃은 이미 여러 번 봤으므로 방법 선택 근거로 쓰지 않음)

사용: python threshold_study.py <reeval 폴더> <이름=학습 xlsx=홀드아웃 프레임 csv> ... <출력 json>
"""
import io
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, sys.argv[1])
from holdout_eval import fit, window_features  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
N_SEEDS, N_REP, N_INNER = 20, 10, 3
RULES = ("pmax", "pmean")
METHODS = ("max", "q99", "avgmax", "avgq99")


def vscore(g, p):
    return pd.DataFrame({"g": g, "p": p}).groupby("g").agg(pmax=("p", "max"), pmean=("p", "mean"))


def oof_scores(X, y, g, n_splits, seed):
    oof = np.zeros(len(y))
    for tr, va in StratifiedGroupKFold(n_splits, shuffle=True, random_state=seed).split(X, y, g):
        oof[va] = fit(X[tr], y[tr]).predict_proba(X[va])[:, 1]
    return vscore(g, oof)


def thresholds(score_list, normal):
    """score_list: 분할 시드별 영상 점수 표. normal: 정상 영상 이름."""
    first = score_list[0].loc[normal]
    avg = sum(s.loc[normal] for s in score_list) / len(score_list)
    out = {}
    for r in RULES:
        out[r, "max"], out[r, "q99"] = first[r].max(), first[r].quantile(0.99)
        out[r, "avgmax"], out[r, "avgq99"] = avg[r].max(), avg[r].quantile(0.99)
    return out


result = {}
for spec in sys.argv[2:-1]:
    name, xlsx, hold_csv = spec.split("=")
    print(f"\n===== {name} =====", flush=True)
    d = pd.read_excel(xlsx)
    cols = [c for c in d.columns if c not in ("filename", "cheat", "start_frame")]
    X, y, g = d[cols].to_numpy(float), d["cheat"].to_numpy(), d["filename"].to_numpy()
    lab = d.groupby("filename")["cheat"].first()
    normal, cheat = lab.index[lab == 0], lab.index[lab == 1]
    res = result[name] = {}

    # 1부: 분할 시드에 따른 임계값 흔들림과, 그 임계값을 넘는 치트 영상 수(OOF 기준)
    seeds = [oof_scores(X, y, g, 5, s) for s in range(N_SEEDS)]
    print(f"[1부] 분할 시드 {N_SEEDS}개. 임계값 최소~최대 / 그 임계값으로 잡히는 학습 치트 영상 수(25개 중) 최소~최대")
    res["part1"] = {}
    for r in RULES:
        single = {m: [thresholds([s], normal)[r, m] for s in seeds] for m in ("max", "q99")}
        groups = [seeds[i:i + 5] for i in range(0, N_SEEDS, 5)]  # 5개씩 평균 -> 4묶음
        single.update({m: [thresholds(gr, normal)[r, m] for gr in groups] for m in ("avgmax", "avgq99")})
        for m in METHODS:
            caught = []
            for i, t in enumerate(single[m]):
                sc = seeds[i] if m in ("max", "q99") else sum(groups[i]) / 5
                caught.append(int((sc.loc[cheat, r] > t).sum()))
            res["part1"][f"{r}_{m}"] = dict(thr_min=float(min(single[m])), thr_max=float(max(single[m])),
                                            caught_min=min(caught), caught_max=max(caught))
            print(f"  {r:5s} {m:7s} 임계값 {min(single[m]):.5f}~{max(single[m]):.5f} | 치트 {min(caught)}~{max(caught)}", flush=True)

    # 2부: 중첩 교차검증
    tp = {(r, m): [] for r in RULES for m in METHODS}
    fp = {(r, m): [] for r in RULES for m in METHODS}
    for rep in range(N_REP):
        ctp = {k: 0 for k in tp}
        cfp = {k: 0 for k in tp}
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(X, y, g):
            inner = [oof_scores(X[tr], y[tr], g[tr], 4, 100 + rep + 1000 * k) for k in range(N_INNER)]
            ntr = np.unique(g[tr][y[tr] == 0])
            th = thresholds(inner, ntr)
            vt = vscore(g[te], fit(X[tr], y[tr]).predict_proba(X[te])[:, 1])
            yv = lab.loc[vt.index].to_numpy()
            for (r, m), t in th.items():
                flag = (vt[r] > t).to_numpy()
                ctp[r, m] += int((flag & (yv == 1)).sum())
                cfp[r, m] += int((flag & (yv == 0)).sum())
        for k in tp:
            tp[k].append(ctp[k])
            fp[k].append(cfp[k])
        print(f"  rep {rep} done", flush=True)
    print(f"[2부] 중첩 교차검증 {N_REP}회. 재현율 평균(반복별 최소~최대, 표준편차) | 오탐률 평균(반복별 최소~최대)")
    res["part2"] = {}
    for k in tp:
        a, b = np.array(tp[k]), np.array(fp[k])
        res["part2"]["_".join(k)] = dict(recall=a.mean() / len(cheat), tp_min=int(a.min()), tp_max=int(a.max()), tp_std=float(a.std()),
                                         fpr=b.mean() / len(normal), fp_min=int(b.min()), fp_max=int(b.max()))
        print(f"  {k[0]:5s} {k[1]:7s} 재현율 {a.mean()/len(cheat):.3f} ({a.min()}~{a.max()}개, sd {a.std():.1f}) | "
              f"오탐률 {b.mean()/len(normal)*100:.2f}% ({b.min()}~{b.max()}개)", flush=True)

    # 3부: 홀드아웃 (참고)
    th = thresholds(seeds[:5], normal)
    hw = window_features(pd.read_csv(hold_csv))
    hv = vscore(hw.filename.to_numpy(), fit(X, y).predict_proba(hw[cols].to_numpy(float))[:, 1])
    hy = hv.index.str.startswith("cheats/")
    print("[3부] 홀드아웃(참고). 임계값 | 적발 | 오탐")
    res["part3"] = {}
    flags = {}
    for (r, m), t in th.items():
        f = (hv[r] > t).to_numpy()
        flags[f"{r}_{m}"] = f
        res["part3"][f"{r}_{m}"] = dict(thr=float(t), tp=int((f & hy).sum()), fp=int((f & ~hy).sum()))
        print(f"  {r:5s} {m:7s} {t:.5f} | {int((f & hy).sum())}/{int(hy.sum())} | {int((f & ~hy).sum())}/{int((~hy).sum())}", flush=True)
    pd.DataFrame(flags, index=hv.index).to_csv(sys.argv[-1].replace(".json", f"_{name}_holdout.csv"))

json.dump(result, open(sys.argv[-1], "w"), indent=2)

# CPU와 GPU의 홀드아웃 판정이 얼마나 일치하는가
names = list(result)
if len(names) == 2:
    a, b = (pd.read_csv(sys.argv[-1].replace(".json", f"_{n}_holdout.csv"), index_col=0) for n in names)
    print(f"\n[{names[0]} vs {names[1]}] 홀드아웃에서 판정이 다른 영상 수")
    for c in a.columns:
        print(f"  {c:14s} {int((a[c] != b.loc[a.index, c]).sum())}")
