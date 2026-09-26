-- Adapted from Project Blackbox (https://github.com/alejandro-publius/blackbox-datahub), Apache-2.0. Modified for Retrace.
CREATE OR REPLACE TABLE marts.exec_metric AS
WITH latest AS (
    SELECT MAX(day) AS kpi_day FROM marts.fct_revenue
),
trailing_stats AS (
    SELECT MEDIAN(f.revenue_usd) AS trailing_median
    FROM marts.fct_revenue AS f, latest AS l
    WHERE f.day BETWEEN l.kpi_day - 35 AND l.kpi_day - 8
)
SELECT
    l.kpi_day,
    f.revenue_usd AS revenue,
    t.trailing_median AS trailing_28d_median_revenue,
    f.revenue_usd / t.trailing_median AS anomaly_ratio
FROM latest AS l
JOIN marts.fct_revenue AS f ON f.day = l.kpi_day
CROSS JOIN trailing_stats AS t;
