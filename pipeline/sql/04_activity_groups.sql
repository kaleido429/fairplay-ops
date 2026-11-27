-- 활동량 구간별 오탐률: 홀드아웃 정상 영상을 평균 속도 순으로 4등분했을 때 구간별 오탐
-- (영상 평균 규칙, 분할 20회 평균 임계값 기준)
WITH ranked AS (
    SELECT d.video_id, d.flagged, a.mean_velocity,
           NTILE(4) OVER (ORDER BY a.mean_velocity) AS quartile   -- 1 = 가장 덜 움직인 25%
    FROM verdicts d
    JOIN video_activity a ON a.video_id = d.video_id
    WHERE d.split = 'holdout' AND d.is_cheat = 0 AND d.rule = 'mean' AND d.method = 'avg20'
)
SELECT quartile,
       COUNT(*)                         AS videos,
       ROUND(MIN(mean_velocity), 2)     AS velocity_min,
       ROUND(MAX(mean_velocity), 2)     AS velocity_max,
       SUM(flagged)                     AS false_positives,
       ROUND(1.0 * SUM(flagged) / COUNT(*), 3) AS fpr
FROM ranked
GROUP BY quartile
ORDER BY quartile;
