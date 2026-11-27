-- 데이터 현황: 구분별 영상 수, 프레임 수, 추출 시간
SELECT split,
       CASE is_cheat WHEN 1 THEN '치트' ELSE '정상' END AS label,
       COUNT(*)                                          AS videos,
       SUM(status = 'extracted')                         AS extracted,
       SUM(status = 'failed')                            AS failed,
       SUM(n_frames)                                     AS frames,
       SUM(fps_fixed)                                    AS fps_fixed,
       SUM(resized_frames > 0)                           AS resized,
       ROUND(SUM(extract_seconds))                       AS extract_seconds
FROM videos
GROUP BY split, is_cheat
ORDER BY split DESC, is_cheat DESC;
