"""영상 폴더에서 판정 결과까지 한 번에 실행.

단계
  ingest  : 영상 폴더를 훑어 videos 테이블에 등록
  extract : Optical Flow로 프레임별 조준 이동량을 계산해 frames 테이블에 저장 (이미 처리한 영상은 건너뜀)
  score   : 윈도우 특징 계산, 모델 학습, 임계값 결정, 윈도우 확률을 windows 테이블에 저장
  report  : sql/ 폴더의 쿼리를 순서대로 실행해 출력

사용
  python run_pipeline.py --db fairplay.db --train <학습 영상 폴더> --holdout <홀드아웃 영상 폴더>
  python run_pipeline.py --db fairplay.db --stages report        # 특정 단계만

영상 폴더 구조: <폴더>/cheats/*.webm (치트), <폴더>/videos/*.webm (정상)
"""
import argparse
import datetime
import glob
import io
import os
import sqlite3
import sys
import time
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
N_SEEDS = 20


def connect(path):
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(open(os.path.join(HERE, "schema.sql"), encoding="utf-8").read())
    return con


def log_run(con, stage, t0, note):
    con.execute("INSERT INTO runs (stage, started, seconds, note) VALUES (?, ?, ?, ?)",
                (stage, datetime.datetime.now().isoformat(timespec="seconds"), round(time.perf_counter() - t0, 1), note))
    con.commit()


def ingest(con, roots):
    t0 = time.perf_counter()
    n = 0
    for split, root in roots.items():
        if not root:
            continue
        for d, _, files in os.walk(root):
            for f in files:
                if not f.endswith(".webm"):
                    continue
                path = os.path.join(d, f)
                rel = os.path.relpath(path, root).replace("\\", "/")
                cur = con.execute(
                    "INSERT OR IGNORE INTO videos (filename, split, is_cheat, path) VALUES (?, ?, ?, ?)",
                    (rel, split, int(rel.startswith("cheats/")), path))
                n += cur.rowcount
    log_run(con, "ingest", t0, f"새 영상 {n}개")
    total = con.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    print(f"[ingest] 새로 등록 {n}개, 전체 {total}개")


def extract(con, backend, n_proc):
    import flow
    if backend == "auto":
        backend = "gpu" if flow.gpu_available() else "cpu"
    todo = con.execute("SELECT video_id, path FROM videos WHERE status != 'extracted' ORDER BY video_id").fetchall()
    print(f"[extract] 방식 {backend}, 대상 {len(todo)}개, 프로세스 {n_proc}개", flush=True)
    if not todo:
        return
    t0 = time.perf_counter()
    ok = 0
    with Pool(n_proc) as pool:
        jobs = [(vid, path, backend) for vid, path in todo]
        for i, (vid, r, err) in enumerate(pool.imap_unordered(flow.extract, jobs), 1):
            if r is None:
                con.execute("UPDATE videos SET status = 'failed' WHERE video_id = ?", (vid,))
                print(f"  [{i}/{len(todo)}] video {vid} 실패: {err}", flush=True)
            else:
                # 영상 하나의 프레임 저장과 상태 변경을 한 트랜잭션으로 묶는다
                con.execute("DELETE FROM frames WHERE video_id = ?", (vid,))
                con.executemany("INSERT INTO frames (video_id, frame, dx, dy) VALUES (?, ?, ?, ?)",
                                [(vid, k, dx, dy) for k, (dx, dy) in enumerate(r["moves"])])
                con.execute(
                    "UPDATE videos SET width = ?, height = ?, fps = ?, fps_fixed = ?, n_frames = ?, resized_frames = ?, "
                    "backend = ?, extract_seconds = ?, status = 'extracted' WHERE video_id = ?",
                    (r["width"], r["height"], r["fps"], r["fps_fixed"], len(r["moves"]), r["resized"], backend,
                     round(r["seconds"], 2), vid))
                ok += 1
                if i % 20 == 0 or i == len(todo):
                    print(f"  [{i}/{len(todo)}] {time.perf_counter() - t0:.0f}초 경과", flush=True)
            con.commit()
    log_run(con, "extract", t0, f"{backend}, 성공 {ok}개, 실패 {len(todo) - ok}개")
    print(f"[extract] 성공 {ok}개, 실패 {len(todo) - ok}개, {time.perf_counter() - t0:.0f}초")


def score(con, n_proc):
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold

    import features

    t0 = time.perf_counter()
    vids = pd.read_sql("SELECT video_id, split, is_cheat FROM videos WHERE status = 'extracted'", con)
    fr = pd.read_sql("SELECT video_id, dx, dy FROM frames ORDER BY video_id, frame", con)
    jobs = [(int(vid), g[["dx", "dy"]].to_numpy()) for vid, g in fr.groupby("video_id")]
    with Pool(n_proc) as pool:
        w = pd.concat(pool.map(features.video_windows, jobs, chunksize=4), ignore_index=True)
    w = w.merge(vids, on="video_id")
    cols = [c for c in w.columns if c not in ("video_id", "start_frame", "split", "is_cheat")]
    tr, ho = w[w.split == "train"], w[w.split == "holdout"]
    X, y, g = tr[cols].to_numpy(float), tr.is_cheat.to_numpy(), tr.video_id.to_numpy()
    print(f"[score] 윈도우 {len(w)}개 (train {len(tr)}, holdout {len(ho)}), 특징 {len(cols)}개, "
          f"{time.perf_counter() - t0:.0f}초", flush=True)

    # 임계값: 학습 영상을 5개 묶음으로 나눠, 각 영상을 '그 영상을 빼고 학습한 모델'로 채점(OOF).
    # 정상 영상 중 최고 점수가 임계값. 분할 시드를 20번 바꿔 영상 점수를 평균한 것이 avg20.
    normal = tr.loc[tr.is_cheat == 0, "video_id"].unique()
    per_seed, oof0 = [], None
    for seed in range(N_SEEDS):
        oof = np.zeros(len(tr))
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, g):
            oof[b] = features.fit(X[a], y[a]).predict_proba(X[b])[:, 1]
        if seed == 0:
            oof0 = oof
        s = pd.DataFrame({"g": g, "p": oof}).groupby("g")["p"].agg(["max", "mean"])
        per_seed.append(s.loc[normal])
    avg = sum(per_seed) / N_SEEDS
    thr = [(rule, "seed0", float(per_seed[0][rule].max())) for rule in ("max", "mean")]
    thr += [(rule, "avg20", float(avg[rule].max())) for rule in ("max", "mean")]

    # 홀드아웃은 학습 영상 전체로 학습한 모델로 채점
    p_ho = features.fit(X, y).predict_proba(ho[cols].to_numpy(float))[:, 1] if len(ho) else []

    con.execute("DELETE FROM windows")
    con.execute("DELETE FROM thresholds")
    con.executemany("INSERT INTO windows (video_id, start_frame, prob) VALUES (?, ?, ?)",
                    [(int(a), int(b), float(c)) for a, b, c in zip(tr.video_id, tr.start_frame, oof0)]
                    + [(int(a), int(b), float(c)) for a, b, c in zip(ho.video_id, ho.start_frame, p_ho)])
    con.executemany("INSERT INTO thresholds (rule, method, value) VALUES (?, ?, ?)", thr)
    con.commit()
    log_run(con, "score", t0, f"윈도우 {len(w)}개, 시드 {N_SEEDS}개")
    for rule, method, value in thr:
        print(f"  임계값 {rule:4s} {method}: {value:.5f}")
    print(f"[score] {time.perf_counter() - t0:.0f}초")


def report(con):
    for path in sorted(glob.glob(os.path.join(HERE, "sql", "*.sql"))):
        sql = open(path, encoding="utf-8").read()
        title = sql.splitlines()[0].lstrip("- ").strip()
        cur = con.execute(sql)
        rows = cur.fetchall()
        head = [d[0] for d in cur.description]
        fmt = [[("" if v is None else f"{v:.5f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)) for v in r]
               for r in rows]
        width = [max(len(h), *(len(r[i]) for r in fmt)) if fmt else len(h) for i, h in enumerate(head)]
        print(f"\n== {os.path.basename(path)}: {title}")
        print("  " + " | ".join(h.ljust(width[i]) for i, h in enumerate(head)))
        for r in fmt:
            print("  " + " | ".join(v.ljust(width[i]) for i, v in enumerate(r)))


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(HERE, "fairplay.db"))
    ap.add_argument("--train")
    ap.add_argument("--holdout")
    ap.add_argument("--backend", default="auto", choices=["auto", "gpu", "cpu"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--stages", default="ingest,extract,score,report")
    a = ap.parse_args()
    con = connect(a.db)
    t0 = time.perf_counter()
    for st in a.stages.split(","):
        if st == "ingest":
            ingest(con, {"train": a.train, "holdout": a.holdout})
        elif st == "extract":
            extract(con, a.backend, a.procs)
        elif st == "score":
            score(con, a.procs)
        elif st == "report":
            report(con)
        else:
            sys.exit(f"알 수 없는 단계: {st}")
    print(f"\n전체 {time.perf_counter() - t0:.0f}초")
