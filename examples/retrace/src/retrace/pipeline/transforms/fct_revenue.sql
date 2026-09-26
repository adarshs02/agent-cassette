-- Adapted from Project Blackbox (https://github.com/alejandro-publius/blackbox-datahub), Apache-2.0. Modified for Retrace.
CREATE OR REPLACE TABLE marts.fct_revenue AS
SELECT
    o.order_day AS day,
    COUNT(*) AS order_count,
    SUM(o.amount * f.usd_rate) AS revenue_usd,
    AVG(o.amount * f.usd_rate) AS aov_usd,
    MEDIAN(o.amount * f.usd_rate) AS aov_median_usd,
    SUM(CASE WHEN c.segment = 'enterprise' THEN o.amount * f.usd_rate ELSE 0 END)
        AS enterprise_revenue_usd
FROM staging.stg_orders AS o
JOIN staging.stg_customers AS c ON c.customer_id = o.customer_id
LEFT JOIN staging.stg_fx_rates AS f
    ON f.rate_day = o.order_day AND f.currency = o.currency
GROUP BY o.order_day
ORDER BY day;
