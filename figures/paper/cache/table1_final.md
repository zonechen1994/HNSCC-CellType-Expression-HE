# Table 1. Cohort characteristics

| Characteristic | TCGA-HNSC | CPTAC-HNSCC | HANCOCK | P value |
|---|---|---|---|---|
| **N patients (clinical)** | 523 | 108 | 763 |  |
| N with survival data | 483 | 105 | 763 |  |
| N slides analysed (WSI) | 442 | 110 | 701 |  |
| Age, median [IQR], yrs | 61 [53-69] | 62 [55-67] | 61 [54-69] | 0.831 |
| Sex, n (%) |  |  |  | <0.001 |
|   Male | 382 (73.0%) | 94 (87.0%) | 614 (80.5%) |  |
|   Female | 141 (27.0%) | 14 (13.0%) | 149 (19.5%) |  |
| T stage, n (%) |  |  |  | <0.001 |
|   T1 | 49 (10.7%) | 9 (8.6%) | 272 (35.9%) |  |
|   T2 | 138 (30.0%) | 39 (37.1%) | 267 (35.2%) |  |
|   T3 | 99 (21.5%) | 28 (26.7%) | 130 (17.2%) |  |
|   T4 | 174 (37.8%) | 29 (27.6%) | 89 (11.7%) |  |
| N stage, n (%) |  |  |  | <0.001 |
|   N0 | 178 (42.0%) | 36 (44.4%) | 200 (32.7%) |  |
|   N1 | 67 (15.8%) | 18 (22.2%) | 113 (18.5%) |  |
|   N2 | 171 (40.3%) | 23 (28.4%) | 246 (40.3%) |  |
|   N3 | 8 (1.9%) | 4 (4.9%) | 52 (8.5%) |  |
| M stage (TCGA only), n (%) |  |  |  |  |
|   M0 | 188 (99.5%) | NA | NA |  |
|   M1 | 1 (0.5%) | NA | NA |  |
| AJCC stage, n (%) |  |  |  | 0.001 |
|   I | 27 (6.0%) | 7 (6.7%) | NA |  |
|   II | 77 (17.0%) | 25 (23.8%) | NA |  |
|   III | 80 (17.7%) | 32 (30.5%) | NA |  |
|   IV | 269 (59.4%) | 41 (39.0%) | NA |  |
| Grade (HANCOCK only), n (%) |  |  |  |  |
|   G1 | NA | NA | 157 (20.6%) |  |
|   G2 | NA | NA | 236 (30.9%) |  |
|   G3 | NA | NA | 370 (48.5%) |  |
| Tumour site (HANCOCK only) | NA | NA | Oropharynx 331; Larynx 207; Oral cavity 134; Hypopharynx 89; CUP 2 |  |
| HPV / p16 status |  |  |  | <0.001 |
|   Positive | 72 (14.8%) | 0 (by design) | 141 (42.5%) |  |
|   Negative | 415 (85.2%) | 108 (by design) | 191 (57.5%) |  |
|   Unknown / not typed | 36 | 0 | 431 |  |
| Median follow-up, months | 20.7 | 24.1 | 39.8 |  |
| Deaths / events | 223 / 483 | 39 / 105 | 213 / 763 |  |


**Notes.** Values are n (%) unless stated otherwise; percentages are computed over cases with the variable recorded (see denominators below), and stage categories collapse subclassifications (T4a/T4b into T4; N2a-c into N2). Age is median [interquartile range]. Follow-up is the time in days divided by 30.44, taken over all cases with survival data. Deaths / events are the number of events over the number of cases with survival data.

**Sources and denominators.** TCGA-HNSC clinical data are from run/TCGA_HNSC/clinical.csv (n=523); T, N, M and AJCC stage from the pathologic staging fields; HPV from the molecular subtype; the model was trained on 442 slides. CPTAC-HNSCC age and sex are from the primary clinical file (n=108); T, N and AJCC stage are from the GDC clinical file (n=105); CPTAC-HNSCC is an HPV-negative cohort by design and has no per-case HPV column. HANCOCK data are from run/HANCOCK/clinical.csv (n=763), with p16/HPV status from the dedicated HPV file (n=332 typed). Sex in HANCOCK was coded 0/1 in the source file and mapped to female/male. Staging with unknown or unassignable values (for example TX, NX, or missing) was excluded from the corresponding percentage and test. Tumour site was not recorded in the TCGA or CPTAC clinical files.

**Statistical tests.** The P value column compares the distribution across the cohorts for which the variable is available, using the Kruskal-Wallis test for age and the chi-squared test for categorical variables (AJCC stage and HPV compare only the two cohorts with those data). These cohorts differ in institution, era, and inclusion criteria, so the comparisons are descriptive of cohort composition and are not intended as a hypothesis test of any biological effect.
