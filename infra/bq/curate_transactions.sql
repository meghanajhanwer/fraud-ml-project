CREATE SCHEMA IF NOT EXISTS `resonant-idea-467410-u9.curated`;

CREATE OR REPLACE TABLE `resonant-idea-467410-u9.curated.bank_transactions_modelready`
PARTITION BY DATE(TransactionDate)
CLUSTER BY AccountID
AS
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
    SAFE_DIVIDE(CAST(TransactionAmount AS FLOAT64), NULLIF(CAST(AccountBalance AS FLOAT64), 0)) AS amt_to_bal,
    EXTRACT(HOUR FROM CAST(TransactionDate AS TIMESTAMP)) AS txn_hour,
    EXTRACT(DAYOFWEEK FROM CAST(TransactionDate AS TIMESTAMP)) - 1 AS txn_dow, 
    TIMESTAMP_DIFF(CAST(TransactionDate AS TIMESTAMP), CAST(PreviousTransactionDate AS TIMESTAMP), MINUTE) AS minutes_since_prev
  FROM `resonant-idea-467410-u9.ingestion.bank_transactions_raw`
),
stats AS (
  SELECT
    APPROX_QUANTILES(TransactionAmount, 100)[OFFSET(90)] AS amount_p90,
    APPROX_QUANTILES(TransactionAmount, 100)[OFFSET(95)] AS amount_p95,
    APPROX_QUANTILES(amt_to_bal, 100)[OFFSET(95)]        AS ratio_p95,
    APPROX_QUANTILES(LoginAttempts, 100)[OFFSET(95)]     AS logins_p95
  FROM base
),

rules AS (
  SELECT
    b.*,
    IF(b.txn_dow IN (0,6), 1, 0) AS is_weekend,
    IF(b.TransactionAmount >= s.amount_p95, 1, 0)                                    AS flag_high_amount,
    IF(b.amt_to_bal >= s.ratio_p95, 1, 0)                                            AS flag_high_ratio,
    IF(b.LoginAttempts >= GREATEST(s.logins_p95, 3), 1, 0)                           AS flag_many_logins,
    IF(b.txn_hour BETWEEN 0 AND 5, 1, 0)                                             AS flag_night,
    IF(b.minutes_since_prev IS NOT NULL
       AND b.minutes_since_prev <= 2
       AND b.TransactionAmount >= s.amount_p90, 1, 0)                                AS flag_rapid_repeat_high,
    IF((b.txn_dow IN (0,6)) AND b.TransactionAmount >= s.amount_p90, 1, 0)           AS flag_weekend_high
  FROM base b
  CROSS JOIN stats s
),

labeled AS (
  SELECT
    r.TransactionID,
    r.AccountID,
    r.TransactionAmount,
    r.TransactionDate,
    r.TransactionType,
    r.Location,
    r.DeviceID,
    r.IP_Address,
    r.MerchantID,
    r.Channel,
    r.CustomerAge,
    r.CustomerOccupation,
    r.TransactionDuration,
    r.LoginAttempts,
    r.AccountBalance,
    r.PreviousTransactionDate,
    r.amt_to_bal,
    r.txn_hour,
    r.is_weekend,
    r.minutes_since_prev,
    r.flag_high_amount,
    r.flag_high_ratio,
    r.flag_many_logins,
    r.flag_night,
    r.flag_rapid_repeat_high,
    r.flag_weekend_high,
    ( r.flag_high_amount
    + r.flag_high_ratio
    + r.flag_many_logins
    + r.flag_night
    + r.flag_rapid_repeat_high
    + r.flag_weekend_high ) AS risk_score,
    CASE
      WHEN ( r.flag_high_amount
           + r.flag_high_ratio
           + r.flag_many_logins
           + r.flag_night
           + r.flag_rapid_repeat_high
           + r.flag_weekend_high ) >= 2
      THEN 1 ELSE 0
    END AS is_fraud,

    ARRAY_TO_STRING(
      ARRAY(
        SELECT part FROM UNNEST([
          r.TransactionType, r.Location, r.Channel, r.MerchantID, r.DeviceID, r.IP_Address
        ]) AS part
        WHERE part IS NOT NULL AND part != ''
      ),
      ' '
    ) AS nlp_text,
    'heuristic_v2' AS label_source
  FROM rules r
)

SELECT * FROM labeled;
