-- Standard SQL (BigQuery)
-- Creates/overwrites curated table and adds a heuristic is_fraud label.
CREATE SCHEMA IF NOT EXISTS `resonant-idea-467410-u9.curated`;

CREATE OR REPLACE TABLE `resonant-idea-467410-u9.curated.bank_transactions_modelready` AS
WITH base AS (
  SELECT
    CAST(TransactionID AS STRING) AS TransactionID,
    CAST(AccountID AS STRING) AS AccountID,
    CAST(TransactionAmount AS FLOAT64) AS TransactionAmount,
    CAST(TransactionDate AS TIMESTAMP) AS TransactionDate,
    TRIM(CAST(TransactionType AS STRING)) AS TransactionType,
    TRIM(CAST(Location AS STRING)) AS Location,
    TRIM(CAST(DeviceID AS STRING)) AS DeviceID,
    TRIM(CAST(IP_Address AS STRING)) AS IP_Address,
    TRIM(CAST(MerchantID AS STRING)) AS MerchantID,
    TRIM(CAST(Channel AS STRING)) AS Channel,
    CAST(CustomerAge AS INT64) AS CustomerAge,
    TRIM(CAST(CustomerOccupation AS STRING)) AS CustomerOccupation,
    CAST(TransactionDuration AS INT64) AS TransactionDuration,
    CAST(LoginAttempts AS INT64) AS LoginAttempts,
    CAST(AccountBalance AS FLOAT64) AS AccountBalance,
    CAST(PreviousTransactionDate AS TIMESTAMP) AS PreviousTransactionDate,

    -- Derived features for heuristics
    SAFE_DIVIDE(CAST(TransactionAmount AS FLOAT64), NULLIF(CAST(AccountBalance AS FLOAT64), 0)) AS amt_to_bal,
    EXTRACT(HOUR FROM CAST(TransactionDate AS TIMESTAMP)) AS txn_hour,
    EXTRACT(DAYOFWEEK FROM CAST(TransactionDate AS TIMESTAMP)) - 1 AS txn_dow, -- 0=Sun..6=Sat
    TIMESTAMP_DIFF(CAST(TransactionDate AS TIMESTAMP), CAST(PreviousTransactionDate AS TIMESTAMP), MINUTE) AS minutes_since_prev
  FROM `resonant-idea-467410-u9.ingestion.bank_transactions_raw`
),
stats AS (
  SELECT
    APPROX_QUANTILES(TransactionAmount, 100)[OFFSET(90)] AS amount_p90,
    APPROX_QUANTILES(TransactionAmount, 100)[OFFSET(95)] AS amount_p95,
    APPROX_QUANTILES(amt_to_bal, 100)[OFFSET(95)]          AS ratio_p95,
    APPROX_QUANTILES(LoginAttempts, 100)[OFFSET(95)]       AS logins_p95
  FROM base
)
SELECT
  b.* EXCEPT(txn_dow),
  -- weekend flag
  IF(b.txn_dow IN (0,6), 1, 0) AS is_weekend,

  -- Heuristic fraud indicators
  IF(b.TransactionAmount >= s.amount_p95, 1, 0)                                    AS flag_high_amount,
  IF(b.amt_to_bal >= s.ratio_p95, 1, 0)                                            AS flag_high_ratio,
  IF(b.LoginAttempts >= GREATEST(s.logins_p95, 3), 1, 0)                           AS flag_many_logins,
  IF(b.txn_hour BETWEEN 0 AND 5, 1, 0)                                             AS flag_night,
  IF(b.minutes_since_prev IS NOT NULL
     AND b.minutes_since_prev <= 2
     AND b.TransactionAmount >= s.amount_p90, 1, 0)                                AS flag_rapid_repeat_high,
  IF((b.txn_dow IN (0,6)) AND b.TransactionAmount >= s.amount_p90, 1, 0)           AS flag_weekend_high,

  -- Risk score & weak label
  (
    IF(b.TransactionAmount >= s.amount_p95, 1, 0) +
    IF(b.amt_to_bal >= s.ratio_p95, 1, 0) +
    IF(b.LoginAttempts >= GREATEST(s.logins_p95, 3), 1, 0) +
    IF(b.txn_hour BETWEEN 0 AND 5, 1, 0) +
    IF(b.minutes_since_prev IS NOT NULL AND b.minutes_since_prev <= 2
       AND b.TransactionAmount >= s.amount_p90, 1, 0) +
    IF((b.txn_dow IN (0,6)) AND b.TransactionAmount >= s.amount_p90, 1, 0)
  ) AS risk_score,

  CASE
    WHEN (
      IF(b.TransactionAmount >= s.amount_p95, 1, 0) +
      IF(b.amt_to_bal >= s.ratio_p95, 1, 0) +
      IF(b.LoginAttempts >= GREATEST(s.logins_p95, 3), 1, 0) +
      IF(b.txn_hour BETWEEN 0 AND 5, 1, 0) +
      IF(b.minutes_since_prev IS NOT NULL AND b.minutes_since_prev <= 2
         AND b.TransactionAmount >= s.amount_p90, 1, 0) +
      IF((b.txn_dow IN (0,6)) AND b.TransactionAmount >= s.amount_p90, 1, 0)
    ) >= 2 THEN 1 ELSE 0
  END AS is_fraud,

  -- Synthesized NLP text from categorical/context fields (skip NULL/empty parts)
  ARRAY_TO_STRING(
    ARRAY(
      SELECT part FROM UNNEST([
        b.TransactionType, b.Location, b.Channel, b.MerchantID, b.DeviceID, b.IP_Address
      ]) AS part
      WHERE part IS NOT NULL AND part != ''
    ),
    ' '
  ) AS nlp_text

FROM base b
CROSS JOIN stats s;
