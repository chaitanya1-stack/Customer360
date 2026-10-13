# RetailLens — RFM Customer Segmentation & Retention Analytics

- Engineered SQL-based RFM and behavioral features from 1.07M retail transaction records using joins, aggregations and window functions, modeling 5,256 historical customers after auditable cleaning.
- Benchmarked K-Means, Ward hierarchical clustering and Gaussian mixture models across 2–8 clusters; selected a 3-segment KMeans solution with 0.323 silhouette and 0.965 mean subsampling ARI.
- Reduced 10 features to 6 principal components (92.9% variance retained), profiled RFM personas and evaluated segment purchase behavior on an untouched 90-day holdout.

Use these only after you have reviewed the code and can explain the methodology. These results measure separation and stability, not classification accuracy or business uplift. '1.07M records' means raw line items, not unique orders. The selected model describes customers observed before 2011-09-10; it does not use the future period for feature fitting.
