"""영상 -> 프레임별 조준 이동량(dx, dy). Farneback Optical Flow, GPU(CUDA) 또는 CPU.

전처리는 Fairplay-Fairy/reeval/extract_aim.py와 같다.
- 시작 3초 제외 (FPS 메타데이터가 비정상이면 30으로 간주)
- 화면 중앙 60% 영역
- 파라미터 0.5 / 3 / 15 / 3 / 5 / 1.2 / 0, 평균 흐름의 반대 방향 = 조준 이동
재생 도중 프레임 크기가 바뀌면 첫 프레임 크기로 맞춘다.

GPU 모드는 CUDA로 빌드한 OpenCV가 필요하다. 위치는 환경변수 OPENCV_CUDA_DIR(기본 D:/opencv-cuda/install).
"""
import os
import shutil
import sys
import tempfile
import time

_cv_dir = os.environ.get("OPENCV_CUDA_DIR", r"D:\opencv-cuda\install")
_cuda_bin = os.environ.get("CUDA_BIN", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4\bin")
if os.path.isdir(os.path.join(_cv_dir, "python")):
    for d in (_cuda_bin, os.path.join(_cv_dir, "x64", "vc17", "bin"), os.path.join(_cv_dir, "bin")):
        if os.path.isdir(d):
            os.add_dll_directory(d)
    sys.path.insert(0, os.path.join(_cv_dir, "python"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

FARNEBACK = dict(pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)


def gpu_available():
    return hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0


def _track(path, backend):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_fps = cap.get(cv2.CAP_PROP_FPS)
    fixed = not (0 < raw_fps <= 120)
    fps = 30 if fixed else int(raw_fps)
    for _ in range(fps * 3):
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

    ok, frame = cap.read()
    if not ok:
        return None
    prev = gray_roi(frame)
    if backend == "gpu":
        far = cv2.cuda.FarnebackOpticalFlow.create(
            numLevels=FARNEBACK["levels"], pyrScale=FARNEBACK["pyr_scale"], fastPyramids=False,
            winSize=FARNEBACK["winsize"], numIters=FARNEBACK["iterations"], polyN=FARNEBACK["poly_n"],
            polySigma=FARNEBACK["poly_sigma"], flags=FARNEBACK["flags"])
        g_prev, g_cur = cv2.cuda.GpuMat(), cv2.cuda.GpuMat()
        g_prev.upload(prev)
    moves = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        roi = gray_roi(frame)
        if backend == "gpu":
            g_cur.upload(roi)
            flow = far.calc(g_prev, g_cur, None).download()
            g_prev, g_cur = g_cur, g_prev
        else:
            flow = cv2.calcOpticalFlowFarneback(prev, roi, None, **FARNEBACK)
            prev = roi
        moves.append((-float(flow[..., 0].mean()), -float(flow[..., 1].mean())))
    cap.release()
    return dict(moves=moves, width=w, height=h, fps=fps, fps_fixed=int(fixed), resized=resized)


def extract(args):
    """멀티프로세스 작업 단위. (video_id, path, backend) -> (video_id, 결과 dict 또는 None, 오류 문자열)"""
    video_id, path, backend = args
    t0 = time.perf_counter()
    try:
        if path.isascii():
            r = _track(path, backend)
        else:  # Windows의 OpenCV는 한글 경로를 못 열어서 임시 폴더로 복사
            with tempfile.TemporaryDirectory() as td:
                tmp = os.path.join(td, "v.webm")
                shutil.copyfile(path, tmp)
                r = _track(tmp, backend)
    except Exception as e:
        return video_id, None, str(e).splitlines()[0][:200]
    if r is None or len(r["moves"]) < 3:
        return video_id, None, "영상을 열 수 없거나 너무 짧음"
    r["seconds"] = time.perf_counter() - t0
    return video_id, r, None
