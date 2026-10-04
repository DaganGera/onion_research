# Dataset cleaning report (src/00_clean_dataset.py)

- photos before: **12260**; removed: **584**; kept: **11676**
- capture-session groups before: 2256; after merging verified re-shots: **915**
- candidate pairs (CLIP >= 0.92 same class, >= 0.94 cross class): 1406308; with >= 8 matches: 433150; verified (>= 10 RANSAC inliers): 79194
- verified pairs that were in DIFFERENT old groups (the re-shots the old grouping missed): **29465**

## Removed, by reason

| reason                                          |   photos |
|:------------------------------------------------|---------:|
| near-identical re-shot (verified, CLIP >= 0.99) |      557 |
| exact duplicate (identical bytes)               |       27 |

## Kept photos per class

| class_name            |   photos |   groups |   largest group |
|:----------------------|---------:|---------:|----------------:|
| healthy red onion     |     3809 |      267 |            2862 |
| healthy white onion   |     3982 |      363 |            1900 |
| unhealthy red onion   |     1915 |      120 |            1628 |
| unhealthy white onion |     1970 |      165 |            1475 |
