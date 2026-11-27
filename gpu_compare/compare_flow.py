"""같은 프레임 쌍에 대해 Farneback Optical Flow를 CPU와 GPU(CUDA)로 각각 계산해 비교.

extract_aim.py와 같은 전처리(시작 3초 스킵, 중앙 60% ROI, 파라미터 0.5/3/15/3/5/1.2/0).
CUDA로 빌드한 OpenCV로 실행해야 함(환경변수 OPENCV_CUDA_DIR).

사용: python compare_flow.py <영상 루트> <출력 csv> [프로세스 수] [영상 목록 txt]
출력: 프레임별 dx/dy (cpu, gpu), 영상별 시간(_time.csv)
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

from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd


def work(args):
    try:
        return _work(args)
    except Exception as e:  # 프레임 크기가 중간에 바뀌는 영상 등
        print("ERROR", args[1], str(e).splitlines()[0][:150], flush=True)
        return args[1], None, None


def _work(args):
    root, rel = args
    cap = cv2.VideoCapture(os.path.join(root, rel))
    if not cap.isOpened():
        return rel, None, None
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = 30 if not (0 < raw_fps <= 120) else int(raw_fps)
    for _ in range(int(fps * 3)):
        if not cap.read()[0]:
            return rel, None, None
    ok, prev = cap.read()
    if not ok:
        return rel, None, None
    x, y, rw, rh = int(w * 0.2), int(h * 0.2), int(w * 0.6), int(h * 0.6)
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)[y:y + rh, x:x + rw]

    far = cv2.cuda.FarnebackOpticalFlow.create(numLevels=3, pyrScale=0.5, fastPyramids=False, winSize=15,
                                               numIters=3, polyN=5, polySigma=1.2, flags=0)
    g_prev, g_cur = cv2.cuda.GpuMat(), cv2.cuda.GpuMat()
    g_prev.upload(np.ascontiguousarray(prev))
    far.calc(g_prev, g_prev, None)  # 워밍업 (커널 로딩 시간 제외)

    rows, t_cpu, t_gpu, t_dec = [], 0.0, 0.0, 0.0
    i = 0
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            break
        roi = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[y:y + rh, x:x + rw])
        t1 = time.perf_counter()
        fc = cv2.calcOpticalFlowFarneback(prev, roi, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        t2 = time.perf_counter()
        g_cur.upload(roi)
        fg = far.calc(g_prev, g_cur, None).download()  # 업로드·다운로드 포함 시간
        t3 = time.perf_counter()
        t_dec += t1 - t0
        t_cpu += t2 - t1
        t_gpu += t3 - t2
        rows.append((rel, i, -float(fc[..., 0].mean()), -float(fc[..., 1].mean()),
                     -float(fg[..., 0].mean()), -float(fg[..., 1].mean()),
                     float(np.abs(fc - fg).mean()), float(np.abs(fc - fg).max())))
        prev = roi
        g_prev, g_cur = g_cur, g_prev
        i += 1
    cap.release()
    return rel, rows, (rel, w, h, i, t_dec, t_cpu, t_gpu)


if __name__ == "__main__":
    root, out = sys.argv[1], sys.argv[2]
    n_proc = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    if len(sys.argv) > 4:
        rels = [l.strip() for l in open(sys.argv[4], encoding="utf-8") if l.strip()]
    else:
        rels = sorted(os.path.relpath(os.path.join(d, f), root).replace("\\", "/")
                      for d, _, fs in os.walk(root) for f in fs if f.endswith(".webm"))
    print("cv2", cv2.__version__, "cuda devices", cv2.cuda.getCudaEnabledDeviceCount(), "videos", len(rels), flush=True)
    frames, times = [], []
    with Pool(n_proc) as pool:
        for k, (rel, rows, t) in enumerate(pool.imap_unordered(work, [(root, r) for r in rels]), 1):
            print(f"[{k}/{len(rels)}] {rel} {'FAILED' if rows is None else t[3:]}", flush=True)
            if rows:
                frames += rows
                times.append(t)
    pd.DataFrame(frames, columns=["filename", "frame", "dx_cpu", "dy_cpu", "dx_gpu", "dy_gpu", "px_absdiff_mean",
                                  "px_absdiff_max"]).sort_values(["filename", "frame"]).to_csv(out, index=False)
    pd.DataFrame(times, columns=["filename", "width", "height", "frames", "t_decode", "t_cpu", "t_gpu"]).sort_values(
        "filename").to_csv(out.replace(".csv", "_time.csv"), index=False)
