-- SQLite 3.25+. :cutoff is an EXCLUSIVE feature boundary; :recent_start = cutoff - 90d.
-- Frequency counts invoices, not line items. Returns remain a separate ledger.
WITH purchases AS (
 SELECT * FROM lines WHERE reason='purchase' AND invoice_date < :cutoff
), orders AS (
 SELECT customer_id,invoice,MIN(invoice_date) AS order_date,
        SUM(value) AS order_value,SUM(quantity) AS basket_units,COUNT(*) AS line_count
 FROM purchases GROUP BY customer_id,invoice
), ordered AS (
 SELECT *,LAG(order_date) OVER (PARTITION BY customer_id ORDER BY order_date,invoice) AS previous_order,
        ROW_NUMBER() OVER (PARTITION BY customer_id ORDER BY order_date,invoice) AS order_sequence
 FROM orders
), order_stats AS (
 SELECT customer_id,COUNT(*) AS frequency,SUM(order_value) AS monetary,
        AVG(order_value) AS avg_order_value,AVG(basket_units) AS avg_basket_units,
        MIN(order_date) AS first_purchase,MAX(order_date) AS last_purchase,
        CAST(julianday(:cutoff)-julianday(date(MAX(order_date))) AS INTEGER) AS recency,
        CAST(julianday(:cutoff)-julianday(date(MIN(order_date))) AS INTEGER) AS customer_age_days,
        CAST(julianday(date(MAX(order_date)))-julianday(date(MIN(order_date))) AS INTEGER) AS purchase_span_days,
        AVG(julianday(order_date)-julianday(previous_order)) AS avg_gap_days,
        COUNT(DISTINCT substr(order_date,1,7)) AS active_months,
        SUM(CASE WHEN order_date >= :recent_start THEN 1 ELSE 0 END)*1.0/COUNT(*) AS recent_order_share
 FROM ordered GROUP BY customer_id
), products AS (
 SELECT customer_id,COUNT(DISTINCT stock_code) AS unique_products,COUNT(*) AS purchase_lines
 FROM purchases GROUP BY customer_id
), refunds AS (
 SELECT customer_id,SUM(value) AS return_value,COUNT(DISTINCT invoice) AS return_invoices
 FROM lines WHERE reason='return' AND invoice_date < :cutoff GROUP BY customer_id
), countries AS (
 SELECT customer_id,country,SUM(value) AS country_value,
        ROW_NUMBER() OVER(PARTITION BY customer_id ORDER BY SUM(value) DESC,country) AS country_rank
 FROM purchases GROUP BY customer_id,country
)
SELECT o.*,p.unique_products,p.purchase_lines,COALESCE(r.return_value,0) AS return_value,
 COALESCE(r.return_invoices,0) AS return_invoices,
 COALESCE(r.return_value,0)/o.monetary AS return_value_ratio,
 o.monetary-COALESCE(r.return_value,0) AS observed_net_value,
 CASE WHEN o.frequency=1 THEN 1 ELSE 0 END AS single_order,
 c.country,substr(o.first_purchase,1,7) AS cohort_month
FROM order_stats o JOIN products p ON o.customer_id=p.customer_id
LEFT JOIN refunds r ON o.customer_id=r.customer_id
LEFT JOIN countries c ON o.customer_id=c.customer_id AND c.country_rank=1
ORDER BY o.customer_id;
