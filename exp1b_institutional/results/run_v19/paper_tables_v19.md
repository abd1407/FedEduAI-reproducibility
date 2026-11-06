# Table 3-D — schedule-length sweep (run_20260910_190700_v19_c668991d)

Pre-specified question: At 50 rounds, what is the directly measured DP cost (A minus D) and does it match the clipping-only control (A minus E, E minus D) as it did at 5-25 rounds?

## Accuracy (%) and AUC under the common threshold policy, by number of rounds (mean ± SD over seeds)

| Rounds | Central accuracy | A: FedAvg accuracy | E: clipping + Poisson accuracy | D: DP + SecAgg (ε = 3) accuracy | Central AUC | A: FedAvg AUC | E: clipping + Poisson AUC | D: DP + SecAgg (ε = 3) AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 89.66 ± 0.20 | 89.49 ± 0.08 | 87.77 ± 0.19 | 87.75 ± 0.22 | 0.9481 ± 0.0012 | 0.9447 ± 0.0015 | 0.9323 ± 0.0015 | 0.9326 ± 0.0016 |
| best validation round | 89.67 ± 0.22 | 89.44 ± 0.19 | 87.67 ± 0.23 | — | 0.9482 ± 0.0011 | 0.9447 ± 0.0015 | 0.9324 ± 0.0015 | — |

## Central − federated gap and corrected equivalence test (common threshold policy)

| Rounds | Accuracy difference | 90% CI | Equivalent (margin 0.01)? | AUC difference | 90% CI | Equivalent (margin 0.005)? |
|---|---:|---|---|---:|---|---|
| 50 | 0.0016 | [-0.0013, 0.0046] | True | 0.0034 | [0.0024, 0.0044] | True |
| best validation round | 0.0023 | [-0.0042, 0.0087] | True | 0.0034 | [0.0018, 0.0051] | False |

## Privacy cost by schedule length (A − D, paired accuracy difference, mean over seeds)

| Rounds | A − D (accuracy) | A − E (accuracy: clipping) | E − D (accuracy: noise) | recorded ε of the maximum client |
|---|---:|---:|---:|---:|
| 50 | 1.74 | 1.73 | 0.02 | 2.999 |

Composed ε over all DP releases in this sweep (maximum college) = 5.47 at δ = 1e-06.