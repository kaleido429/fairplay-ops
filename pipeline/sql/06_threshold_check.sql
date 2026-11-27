-- 임계값 검산: Python이 저장한 seed0 임계값이 SQL로 직접 구한 '학습 정상 영상 최고 점수'와 같은지
SELECT t.rule,
       ROUND(t.value, 6)                                              AS stored_threshold,
       ROUND(CASE t.rule WHEN 'max' THEN s.top_pmax ELSE s.top_pmean END, 6) AS sql_threshold,
       ABS(t.value - CASE t.rule WHEN 'max' THEN s.top_pmax ELSE s.top_pmean END) < 1e-9 AS same
FROM thresholds t
CROSS JOIN (
    SELECT MAX(pmax) AS top_pmax, MAX(pmean) AS top_pmean
    FROM video_scores
    WHERE split = 'train' AND is_cheat = 0
) s
WHERE t.method = 'seed0'
ORDER BY t.rule;
