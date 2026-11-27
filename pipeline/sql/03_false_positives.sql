-- 홀드아웃 오탐 영상과 활동량: 정상인데 치트로 판정된 영상이 '거의 움직이지 않은 영상'인지 확인
-- (영상 평균 규칙, 분할 20회 평균 임계값 기준)
SELECT v.filename,
       v.width || 'x' || v.height                                                AS resolution,
       ROUND(d.score, 3)                                                         AS pmean,
       ROUND(a.mean_velocity, 2)                                                 AS mean_velocity,
       ROUND(a.still_ratio, 2)                                                   AS still_ratio,
       -- 이 영상보다 평균 속도가 낮은 홀드아웃 정상 영상의 비율 (0에 가까울수록 가장 덜 움직인 축)
       ROUND(PERCENT_RANK() OVER (ORDER BY a.mean_velocity), 2)                  AS velocity_rank,
       d.flagged
FROM verdicts d
JOIN videos v         ON v.video_id = d.video_id
JOIN video_activity a ON a.video_id = d.video_id
WHERE d.split = 'holdout' AND d.is_cheat = 0 AND d.rule = 'mean' AND d.method = 'avg20'
ORDER BY d.flagged DESC, d.score DESC
LIMIT 10;
