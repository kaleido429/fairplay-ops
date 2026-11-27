-- 판정 성능: 규칙과 임계값 방식별 적발 수(TP), 오탐 수(FP), 재현율, 오탐률
-- train의 점수는 그 영상을 빼고 학습한 모델의 점수(OOF)라서, 임계값을 정한 데이터에서 본 낙관적인 값이다.
SELECT split, rule, method,
       ROUND(MAX(threshold), 5)                                              AS threshold,
       SUM(is_cheat = 1 AND flagged = 1)                                     AS tp,
       SUM(is_cheat = 1)                                                     AS cheats,
       SUM(is_cheat = 0 AND flagged = 1)                                     AS fp,
       SUM(is_cheat = 0)                                                     AS normals,
       ROUND(1.0 * SUM(is_cheat = 1 AND flagged = 1) / SUM(is_cheat = 1), 3) AS recall,
       ROUND(1.0 * SUM(is_cheat = 0 AND flagged = 1) / SUM(is_cheat = 0), 4) AS fpr
FROM verdicts
GROUP BY split, rule, method
ORDER BY split, rule, method DESC;
