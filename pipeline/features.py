"""프레임별 (dx, dy) -> 프레임 특징 7개 -> 150프레임 윈도우 통계 특징 105개, 그리고 모델.

Fairplay-Fairy/reeval의 extract_aim.per_frame, holdout_eval.window_features / fit과 같은 계산.
윈도우 통계(최빈값, 왜도, 첨도, 엔트로피, 상관계수 등)는 SQL로 쓰기 어려워 Python에서 계산한다.
"""
import itertools

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import entropy

WINDOW, STEP, BINS = 150, 30, 10
RAW = ["dx", "dy", "velocity", "acceleration", "jerk", "angle_deg", "angle_change"]
PARAMS = dict(
    objective="binary:logistic", eval_metric="auc", n_jobs=-1, random_state=42,
    colsample_bytree=0.987032220608099, gamma=0.27398594162404366,
    learning_rate=0.09469418846162371, max_depth=5, min_child_weight=1,
    n_estimators=298, subsample=0.7650470707645706,
)


def per_frame(moves):
    m = np.asarray(moves, dtype=float)
    dx, dy = m[:, 0], m[:, 1]
    acc = np.concatenate([[np.nan], np.linalg.norm(np.diff(m, axis=0), axis=1)])
    jerk = np.concatenate([[np.nan, np.nan], np.linalg.norm(np.diff(m, axis=0, n=2), axis=1)])
    ang = np.arctan2(dy, dx) * 180 / np.pi
    dang = np.concatenate([[np.nan], np.abs((np.diff(ang) + 180) % 360 - 180)])
    return pd.DataFrame(dict(frame=np.arange(len(m)), dx=dx, dy=dy, velocity=np.linalg.norm(m, axis=1),
                             acceleration=acc, jerk=jerk, angle_deg=ang, angle_change=dang))


def window_features(vd):
    """영상 1개의 프레임 특징 표 -> 윈도우 특징 표 (start_frame + 105개)."""
    rows = []
    for i in range(0, len(vd) - WINDOW + 1, STEP):
        ch = vd.iloc[i:i + WINDOW]
        r = {"start_frame": i}
        for c in RAW:
            s = ch[c]
            mode = s.mode()
            r[f"{c}_mean"], r[f"{c}_median"] = s.mean(), s.median()
            r[f"{c}_mode"] = mode.iloc[0] if not mode.empty else np.nan
            r[f"{c}_range"], r[f"{c}_std"] = s.max() - s.min(), s.std()
            r[f"{c}_iqr"] = s.quantile(0.75) - s.quantile(0.25)
            r[f"{c}_max"], r[f"{c}_min"] = s.max(), s.min()
            r[f"{c}_skewness"], r[f"{c}_kurtosis"] = s.skew(), s.kurtosis()
            v = s.values
            r[f"{c}_zcr"] = ((v[:-1] * v[1:]) < 0).sum() / WINDOW
            r[f"{c}_entropy"] = entropy(pd.cut(s, bins=BINS, labels=False).value_counts(normalize=True))
        for a, b in itertools.combinations(RAW, 2):
            k = ch[a].corr(ch[b])
            r[f"corr_{a}_{b}"] = 0 if pd.isna(k) else k
        rows.append(r)
    return pd.DataFrame(rows)


def video_windows(args):
    """멀티프로세스 작업 단위. (video_id, [(dx, dy), ...]) -> video_id가 붙은 윈도우 특징 표"""
    video_id, moves = args
    w = window_features(per_frame(moves))
    w.insert(0, "video_id", video_id)
    return w


def fit(X, y):
    return xgb.XGBClassifier(scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1), **PARAMS).fit(X, y)
