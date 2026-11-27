"""extract_aim.py와 같은 전처리로 프레임별 에임 특징 CSV를 만들되, Optical Flow를 GPU(CUDA)로 계산.

CUDA로 빌드한 OpenCV 필요(환경변수 OPENCV_CUDA_DIR).
Fairplay-Fairy 저장소의 reeval 폴더 위치는 환경변수 REEVAL_DIR(기본: 이 파일 옆의 reeval_copy).
재생 도중 프레임 크기가 바뀌는 영상은 첫 프레임 크기로 맞춘 뒤 계산하고 meta의 resized 열에 프레임 수를 기록.

사용: python extract_aim_gpu.py <영상 루트> <출력 csv> [프로세스 수] [gpu|cpu] [영상 목록 txt]
"""
import os
import sys
import time

# CUDA로 빌드한 OpenCV 위치는 환경변수 OPENCV_CUDA_DIR, CUDA 런타임 위치는 CUDA_BIN으로 지정
_cv = os.environ.get("OPENCV_CUDA_DIR", r"D:\opencv-cuda\install")
os.add_dll_directory(os.environ.get("CUDA_BIN", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin"))
for d in (os.path.join(_cv, "x64", "vc17", "bin"), os.path.join(_cv, "bin")):
    if os.path.isdir(d):
        os.add_dll_directory(d)
sys.path.insert(0, os.path.join(_cv, "python"))
sys.path.insert(1, os.environ.get("REEVAL_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "reeval_copy")))

import shutil
import tempfile
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd
from extract_aim import per_frame


def track(path, mode):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fixed = not (0 < raw_fps <= 120)
    fps = 30 if fixed else int(raw_fps)
    for _ in range(int(fps * 3)):
        if not cap.read()[0]:
            return None
    x, y, rw, rh = int(w * 0.2), int(h * 0.2), int(w * 0.6), int(h * 0.6)
    resized = 0

    def gray_roi(frame):
        nonlocal resized
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))
            resized += 1
        return np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y + rh, x:x + rw])

    ok, prev = cap.read()
    if not ok:
        return None
    prev = gray_roi(prev)
    if mode == "gpu":
        far = cv2.cuda.FarnebackOpticalFlow.create(numLevels=3, pyrScale=0.5, fastPyramids=False, winSize=15,
                                                   numIters=3, polyN=5, polySigma=1.2, flags=0)
        g_prev, g_cur = cv2.cuda.GpuMat(), cv2.cuda.GpuMat()
        g_prev.upload(prev)
    moves = []
    t0 = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        roi = gray_roi(frame)
        if mode == "gpu":
            g_cur.upload(roi)
            flow = far.calc(g_prev, g_cur, None).download()
            g_prev, g_cur = g_cur, g_prev
        else:
            flow = cv2.calcOpticalFlowFarneback(prev, roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            prev = roi
        moves.append((-float(flow[..., 0].mean()), -float(flow[..., 1].mean())))
    cap.release()
    return moves, (w, h, fps, raw_fps, fixed, len(moves), resized, round(time.perf_counter() - t0, 2))


def work(args):
    root, rel, mode = args
    path = os.path.join(root, rel)
    try:
        if path.isascii():
            r = track(path, mode)
        else:
            with tempfile.TemporaryDirectory() as td:
                tmp = os.path.join(td, "v.webm")
                shutil.copyfile(path, tmp)
                r = track(tmp, mode)
    except Exception as e:
        print("ERROR", rel, str(e).splitlines()[0][:150], flush=True)
        return rel, None, None
    if r is None or len(r[0]) < 3:
        return rel, None, None
    return rel, per_frame(rel, r[0]), r[1]


if __name__ == "__main__":
    root, out = sys.argv[1], sys.argv[2]
    n_proc = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    mode = sys.argv[4] if len(sys.argv) > 4 else "gpu"
    if len(sys.argv) > 5:
        rels = [l.strip() for l in open(sys.argv[5], encoding="utf-8") if l.strip()]
    else:
        rels = sorted(os.path.relpath(os.path.join(d, f), root).replace("\\", "/")
                      for d, _, fs in os.walk(root) for f in fs if f.endswith(".webm"))
    print("cv2", cv2.__version__, "mode", mode, "videos", len(rels), flush=True)
    frames, meta = [], []
    t0 = time.perf_counter()
    with Pool(n_proc) as pool:
        for i, (rel, df, info) in enumerate(pool.imap_unordered(work, [(root, r, mode) for r in rels]), 1):
            print(f"[{i}/{len(rels)}] {rel} {'FAILED' if df is None else info}", flush=True)
            if df is not None:
                frames.append(df)
                meta.append((rel, *info))
    print(f"total {time.perf_counter() - t0:.0f}s", flush=True)
    pd.concat(frames).sort_values(["filename", "frame"]).to_csv(out, index=False)
    pd.DataFrame(meta, columns=["filename", "width", "height", "fps", "raw_fps", "fps_fixed", "frames", "resized",
                                "seconds"]).sort_values("filename").to_csv(out.replace(".csv", "_meta.csv"), index=False)
