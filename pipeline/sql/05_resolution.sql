-- 해상도별 현황: 치트 영상과 정상 영상의 해상도 분포, 판정 결과, 프레임당 추출 시간
-- (녹화 환경 편향 확인용. 영상 평균 규칙, 분할 20회 평균 임계값, 홀드아웃)
SELECT v.width || 'x' || v.height                         AS resolution,
       SUM(v.is_cheat = 1)                                AS cheats,
       SUM(v.is_cheat = 1 AND d.flagged = 1)              AS cheats_flagged,
       SUM(v.is_cheat = 0)                                AS normals,
       SUM(v.is_cheat = 0 AND d.flagged = 1)              AS normals_flagged,
       ROUND(1000.0 * SUM(v.extract_seconds) / SUM(v.n_frames), 1) AS ms_per_frame
FROM videos v
JOIN verdicts d ON d.video_id = v.video_id AND d.rule = 'mean' AND d.method = 'avg20'
WHERE v.split = 'holdout'
GROUP BY v.width, v.height
ORDER BY v.width * v.height;
