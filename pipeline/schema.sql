-- Fairplay Fairy 파이프라인 DB 구조 (SQLite)
-- 영상 파일 자체는 DB에 넣지 않는다. DB에는 영상에서 뽑아낸 값만 저장한다.

-- 1) 영상 1개 = 1행. 영상의 기본 정보와 추출 상태.
CREATE TABLE IF NOT EXISTS videos (
    video_id        INTEGER PRIMARY KEY,
    filename        TEXT NOT NULL UNIQUE,   -- 예: cheats/xxxx.webm
    split           TEXT NOT NULL CHECK (split IN ('train', 'holdout')),
    is_cheat        INTEGER NOT NULL CHECK (is_cheat IN (0, 1)),
    path            TEXT NOT NULL,          -- 로컬 파일 위치
    width           INTEGER,
    height          INTEGER,
    fps             INTEGER,
    fps_fixed       INTEGER,                -- FPS 메타데이터가 비정상이라 30으로 간주했으면 1
    n_frames        INTEGER,
    resized_frames  INTEGER,                -- 재생 중 크기가 바뀌어 첫 프레임 크기로 맞춘 프레임 수
    backend         TEXT,                   -- 'gpu' 또는 'cpu'
    extract_seconds REAL,
    status          TEXT NOT NULL DEFAULT 'new'  -- new / extracted / failed
);

-- 2) 프레임 1개 = 1행. Optical Flow로 구한 조준 이동량.
CREATE TABLE IF NOT EXISTS frames (
    video_id INTEGER NOT NULL REFERENCES videos(video_id),
    frame    INTEGER NOT NULL,
    dx       REAL NOT NULL,
    dy       REAL NOT NULL,
    PRIMARY KEY (video_id, frame)
) WITHOUT ROWID;

-- 3) 윈도우(150프레임) 1개 = 1행. 모델이 낸 치트 확률.
--    train 영상은 그 영상을 빼고 학습한 모델의 확률(OOF), holdout 영상은 train 전체로 학습한 모델의 확률.
CREATE TABLE IF NOT EXISTS windows (
    video_id    INTEGER NOT NULL REFERENCES videos(video_id),
    start_frame INTEGER NOT NULL,
    prob        REAL NOT NULL,
    PRIMARY KEY (video_id, start_frame)
) WITHOUT ROWID;

-- 4) 판정 임계값. rule: 'max'(윈도우 최댓값) / 'mean'(영상 평균)
CREATE TABLE IF NOT EXISTS thresholds (
    rule   TEXT NOT NULL,
    method TEXT NOT NULL,   -- 'seed0'(분할 1회) / 'avg20'(분할 20회 평균)
    value  REAL NOT NULL,
    PRIMARY KEY (rule, method)
);

-- 5) 실행 기록
CREATE TABLE IF NOT EXISTS runs (
    run_id   INTEGER PRIMARY KEY,
    stage    TEXT NOT NULL,
    started  TEXT NOT NULL,
    seconds  REAL,
    note     TEXT
);

-- ---------- 뷰: 테이블에서 계산해 보여주는 가상 표 ----------

-- 프레임별 특징. 이전 프레임 값은 LAG로 가져온다.
CREATE VIEW IF NOT EXISTS frame_features AS
WITH d AS (
    SELECT video_id, frame, dx, dy,
           LAG(dx, 1) OVER w AS dx1, LAG(dy, 1) OVER w AS dy1,
           LAG(dx, 2) OVER w AS dx2, LAG(dy, 2) OVER w AS dy2
    FROM frames
    WINDOW w AS (PARTITION BY video_id ORDER BY frame)
), a AS (
    SELECT *,
           SQRT(dx * dx + dy * dy)                                   AS velocity,
           SQRT((dx - dx1) * (dx - dx1) + (dy - dy1) * (dy - dy1))   AS acceleration,
           SQRT((dx - 2 * dx1 + dx2) * (dx - 2 * dx1 + dx2)
              + (dy - 2 * dy1 + dy2) * (dy - 2 * dy1 + dy2))         AS jerk,
           DEGREES(ATAN2(dy, dx))                                    AS angle_deg,
           DEGREES(ATAN2(dy, dx)) - DEGREES(ATAN2(dy1, dx1))         AS dang
    FROM d
)
SELECT video_id, frame, dx, dy, velocity, acceleration, jerk, angle_deg,
       -- 각도 차이를 0~180도로 접는다 (350도와 10도의 차이는 20도)
       ABS((dang + 180) - 360 * FLOOR((dang + 180) / 360) - 180) AS angle_change
FROM a;

-- 영상별 점수: 윈도우 확률의 최댓값과 평균
CREATE VIEW IF NOT EXISTS video_scores AS
SELECT v.video_id, v.filename, v.split, v.is_cheat,
       COUNT(*)    AS n_windows,
       MAX(w.prob) AS pmax,
       AVG(w.prob) AS pmean
FROM videos v
JOIN windows w ON w.video_id = v.video_id
GROUP BY v.video_id;

-- 영상별 판정: 점수가 임계값을 넘으면 flagged = 1. 규칙 2개 × 임계값 방식 2개 = 영상당 4행.
CREATE VIEW IF NOT EXISTS verdicts AS
SELECT s.video_id, s.filename, s.split, s.is_cheat, t.rule, t.method, t.value AS threshold,
       CASE t.rule WHEN 'max' THEN s.pmax ELSE s.pmean END AS score,
       CASE WHEN (CASE t.rule WHEN 'max' THEN s.pmax ELSE s.pmean END) > t.value THEN 1 ELSE 0 END AS flagged
FROM video_scores s
CROSS JOIN thresholds t;

-- 영상별 활동량: 거의 움직이지 않은 영상을 찾기 위한 요약
CREATE VIEW IF NOT EXISTS video_activity AS
SELECT video_id,
       AVG(velocity)                                        AS mean_velocity,
       AVG(CASE WHEN velocity < 0.1 THEN 1.0 ELSE 0.0 END)  AS still_ratio   -- 속도 0.1 미만 프레임 비율
FROM frame_features
GROUP BY video_id;
