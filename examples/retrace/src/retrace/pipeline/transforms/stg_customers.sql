-- Adapted from Project Blackbox (https://github.com/alejandro-publius/blackbox-datahub), Apache-2.0. Modified for Retrace.
CREATE OR REPLACE TABLE staging.stg_customers AS
SELECT
    customer_id,
    segment,
    country,
    CAST(created_at AS DATE) AS created_at
FROM raw.raw_customers;
