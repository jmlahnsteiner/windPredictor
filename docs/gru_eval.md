# GRU vs RF Evaluation Report

N training samples: 3898  |  CV: 5-fold TimeSeriesSplit


| Metric | RF (NWP-enriched) | GRU |
|--------|-------------------|-----|
| roc_auc | 0.827 | 0.668 |
| precision | 0.326 | 0.348 |
| recall | 0.420 | 0.401 |
| f1 | 0.348 | 0.255 |

**RF remains the recommended production model.**
