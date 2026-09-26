-- Adapted from Project Blackbox (https://github.com/alejandro-publius/blackbox-datahub), Apache-2.0. Modified for Retrace.
CREATE OR REPLACE TABLE staging.stg_fx_rates AS
WITH rates AS (
    SELECT
        CAST(rate_day AS DATE) AS rate_day,
        currency,
        CAST(usd_rate AS DOUBLE) AS usd_rate
    FROM raw.raw_fx_rates
),
spine AS (
    SELECT CAST(t.d AS DATE) AS day, c.currency
    FROM range(DATE '2026-05-12', DATE '2026-08-10', INTERVAL 1 DAY) AS t(d)
    CROSS JOIN (SELECT DISTINCT currency FROM rates) AS c
)
SELECT
    s.day AS rate_day,
    s.currency,
    CASE WHEN s.day - r.rate_day <= 2 THEN r.usd_rate END AS usd_rate,
    r.rate_day AS source_rate_day
FROM spine AS s
ASOF LEFT JOIN rates AS r
    ON s.currency = r.currency AND s.day >= r.rate_day;
