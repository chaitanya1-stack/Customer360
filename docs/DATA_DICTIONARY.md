# Data dictionary and assumptions

## Original workbook

Two sheets: Year 2009–2010 and Year 2010–2011. Columns are Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID and Country. Each row is an invoice line item. Timestamps are used as provided, with no inferred timezone. Unit prices are GBP.

## Modeling fields

All values are computed before the exclusive snapshot cutoff. No purchase in the future diagnostic window contributes to these fields.

| Feature | Definition | Treatment |
|---|---|---|
| recency | Calendar days between cutoff and latest purchase date | 1/99% clip, log1p, standardize |
| frequency | Number of distinct retained purchase invoices per customer | 1/99% clip, log1p, standardize |
| monetary | Sum of positive quantity × positive price, merchandise only | 1/99% clip, log1p, standardize |
| unique_products | Distinct merchandise StockCode values | 1/99% clip, log1p, standardize |
| avg_basket_units | Mean sum of units per invoice | 1/99% clip, log1p, standardize |
| customer_age_days | Days from first observed purchase date to cutoff | 1/99% clip, log1p, standardize |
| avg_gap_days | Mean elapsed time between consecutive distinct invoices, from LAG | Median imputation, 1/99% clip, log1p, standardize |
| return_value_ratio | Magnitude of observed return value / gross purchase value | Clip to training observed range, log1p, standardize |
| recent_order_share | Invoices in last 90 days / historical invoices | Clip [0,1], standardize |
| single_order | 1 when frequency is one, otherwise 0 | Preserve flag, standardize |

Historical-only fitting applies to every transform. Exact clipping thresholds, imputation medians and scaling parameters live in the saved model. `avg_gap_days` is missing for single-order customers; missingness is retained via `single_order`. Recency is not log-normal by assumption: the transform is chosen to reduce skew and relative scale.

## Descriptive and identifier fields

| Field | Meaning |
|---|---|
| customer_id | Source identifier, treated as text, excluded from modeling |
| first_purchase / last_purchase | First/last retained historical purchase timestamp |
| avg_order_value | Historical gross purchase value / frequency; descriptive only |
| purchase_span_days | Calendar days between first and last historical purchase |
| active_months | Number of calendar months with ≥1 purchase |
| purchase_lines | Historical retained invoice lines, distinct from frequency |
| return_value / return_invoices | Magnitude of observed returns and number of return invoices |
| observed_net_value | Gross purchase value less return magnitude; not an accounting reconciliation |
| country | Country with largest historical customer gross value; alphabetical tie-break |
| cohort_month | Month of first observed purchase, not verified acquisition |
| cluster | Model-assigned numeric cluster ID |
| rfm_persona | Explicit historical RFM rule label; distinct from cluster |
| PC1 ... PC6 | Scores in the fitted PCA feature space |

## Future-period fields

`future_orders`: positive merchandise invoices in the holdout. `future_gross_value`: their gross spend. `purchased_in_holdout`: 1 if at least one such purchase. Customers with no matching future purchases receive zeros. New future customers are excluded from historical-customer evaluation. Cohort blanks are unobserved future months, whereas zero is an observed month with no returning customer.

## Modeling caveats

RFM and behavior fields are correlated; PCA rotates that geometry rather than making all concepts equally important. No margin, lifetime value, churn label, promotion exposure, quantity discount or product category hierarchy is available. Five-digit product codes with up to two letters are retained; other codes are treated as administrative/non-merchandise. Review this explicit code-pattern assumption before applying the project elsewhere. Returns are not matched to specific original invoices. Deduplication may remove legitimate identical invoice lines; the raw archive and removed count remain auditable.
