# FedEduAI: A Privacy-Aware Federated Multimodal Architecture for Adaptive Learning Analytics and Decision Support

**Code and results package accompanying the manuscript submitted to *PeerJ Computer Science*.**

Authors: Abdullah Alsubhy, Fahad Alqurashi, Adnan Abi Sen, Rashid Mehmood

Zenodo: https://doi.org/10.5281/zenodo.23953300 (record of release v1.0; every release receives its own version DOI and the Zenodo record lists all versions). This is release v1.1 (6 November 2026).

## Summary

FedEduAI is a privacy-aware federated architecture for learning analytics that keeps student data inside each
institution (or faculty) and trains a shared model with federated averaging, secure aggregation (Flower SecAgg+)
and student-level differential privacy (Opacus DP-SGD), while fusing several evidence channels (academic records,
behavioural/affective signals, knowledge-progress assessment and a synthetic multimodal cohort) into a single
decision-support score. This repository contains the notebooks, the generated Flower/Opacus application, the
aggregate result files and the synthetic-data generator that produce the tables and figures reported in the paper.
It is organised by experiment so that every number in the paper can be traced to a notebook cell and a result file.

## Folder map

| Folder | Experiment | Contents |
|---|---|---|
| `exp1a_uci_federated/` | Exp 1-A: federated vs. centralised learning on the public UCI Student Performance data | `FedEduAI_Exp1A_UCI_v25.ipynb` (with outputs: per-seed metrics and the summary table) |
| `exp1b_institutional/` | Exp 1-B: institutional cohort, Flower + Opacus (FedAvg, SecAgg+, DP, controls, LiRA, TOST, schedule sweep) | `FedEduAI_Validation_v16.ipynb` (5-round primary run behind the paper tables), `FedEduAI_Validation_v19.ipynb` (50-round schedule sweep), `flower_app/` (generated Flower application), `results/run_v16/`, `results/run_v19/` |
| `exp2_hybrid_emo/` | Exp 2: Hybrid-Emo facial-expression and drowsiness channel (FER2013; Haar cascade + dlib 68 landmarks + CLIP zero-shot boost; eye-aspect-ratio drowsiness) | **Available on request from the corresponding author** (see `exp2_hybrid_emo/README.md`) |
| `exp3_semantic_knowledge/` | Exp 3: short-answer grading on the Mohler ASAG corpus (text-matching channel + LLM channel) | `FedEduAI_Exp3_LLM_v21.ipynb`, `FedEduAI_Validation_v13.ipynb` (text-matching channel), `results/` (LLM scores + summary, semantic-matching results, alignment audit) |
| `exp4_fusion/` | Exp 4: multimodal fusion on a fully synthetic cohort | `exp4_fusion_generator.py`, `exp4_cohort.csv`, `exp4_results.json`, `exp4_results.md`, `make_fig6_ieee_style.py`, `fig6.png` |

Root files: `README.md`, `LICENSE` (MIT), `CITATION.cff`, `requirements.txt`, `.gitignore`,
`CHECKSUMS.sha256` (SHA-256 of every file in the package).

## How to run each experiment

All notebooks were executed on Google Colab (Python 3.13 runtime). Cell outputs are kept as the evidence behind the
paper; a fresh run re-creates them. Paths of the form `/content/drive/MyDrive/...` are Colab Drive paths and should be
adapted to your environment.

### Exp 1-A: UCI Student Performance (public data)

```
jupyter notebook exp1a_uci_federated/FedEduAI_Exp1A_UCI_v25.ipynb
```

The notebook downloads the UCI archive (id 320) itself, builds the binary pass target (G3 >= 10), partitions the 1,044
students into four natural non-IID clients (school x subject), and trains random forest, XGBoost, ANN, LSTM
(grade sequence), a centralised deep model and the federated deep model (25 rounds, 4 local epochs, FedAvg, seeds 0-4).
Runs on CPU in a few minutes.

### Exp 1-B: institutional cohort (confidential data, not included)

1. Obtain the institutional tables (see *Data statement*). Place them where `CONFIG['paths']` in cell 2 of the
   notebook points (students, grades, absence tables).
2. Run `exp1b_institutional/FedEduAI_Validation_v16.ipynb` top to bottom in a fresh session (`RUN_MODE='full'`).
   Cell 8 writes the Flower/Opacus application (the same files as `flower_app/`), cell 12 trains every arm from
   scratch for seeds 0-4 and writes the JSON result files, cell 13 renders the paper tables (`paper_tables_v16.md`).
3. Run `FedEduAI_Validation_v19.ipynb` for the 50-round schedule sweep (`schedule_sweep_v19.json`, `paper_tables_v19.md`).

Settings of the primary run (from `results/run_v16/run_manifest_v16.json`): 12 clients (faculties with >= 300
students), 5 rounds, 1 local epoch, batch size 64, SGD with learning rate 0.05, clipping norm 1.0, delta = 1e-6,
nominal epsilon in {3, 8}, seeds 0-4, RDP accountant, Poisson sampling. Arms: Central, A FedAvg, B FedAvg + SecAgg+,
C DP (eps 3 / 8), D DP + SecAgg+ (eps 3 / 8), E clipping + Poisson without noise, F Poisson without clipping.
The `flower_app/` directory is a complete Flower 1.22 app (`pyproject.toml`, `fededuai_privacy/`):

```
cd exp1b_institutional/flower_app
pip install -e .                      # flwr==1.22.0, opacus==1.5.4
export FEDEDUAI_DATA_DIR=/path/to/private/shards   # client_<partition>.npz files, never shipped with the app
export FEDEDUAI_SERVER_OUTPUT=/path/to/private/output
flwr run .                            # local simulation; edit [tool.flwr.app.config] for dp / secagg / epsilon / seed
```

`start_server.sh`, `start_client.sh` and `provision_pilot.py` are provided for a TLS-authenticated multi-host pilot;
`monitor_command.py` records wall time, memory and interface bytes of a command. The experiments in the paper used the
single-process transport in `local_grid.py`, which serialises real Flower SecAgg+ messages but does not isolate hosts
(see the warnings inside the result files).

### Exp 2: Hybrid-Emo (FER2013)

The facial-expression / drowsiness code (Experiment 2) is not part of this release; it is available from the
corresponding author on reasonable request (see `exp2_hybrid_emo/README.md`). The pipeline described in the paper uses OpenCV Haar
cascades for face detection, dlib's 68-point landmark model (`shape_predictor_68_face_landmarks.dat`), a CLIP
zero-shot boost for the expression classes and the eye-aspect ratio for drowsiness.

### Exp 3: short-answer grading (Mohler corpus)

* LLM channel: `exp3_semantic_knowledge/FedEduAI_Exp3_LLM_v21.ipynb` (Colab T4 GPU, 40-70 min). Loads the cleaned
  Mohler corpus from Hugging Face (`nkazi/MohlerASAG`), rescales the two 0-10 assignments to 0-5, grades the 2,442
  answers with Qwen2.5-7B-Instruct (4-bit NF4, greedy decoding, max 6 new tokens) and writes
  `llm_scores_v21_Qwen2.5-7B-Instruct.csv` and `llm_summary_v21_Qwen2.5-7B-Instruct.json`
  (RMSE 1.088, MAE 0.909, Pearson 0.720, Spearman 0.683 against the mean of the two graders).
* Text-matching (TM) channel (TF-IDF / ROUGE / Jaccard features with a Ridge regressor) and the TM + LLM
  combination: this code lives inside the institutional validation notebook of the v13 run
  (`FedEduAI_Validation_v13.ipynb`, Exp 3 section), not in a stand-alone notebook. Its outputs are
  `results/exp3_semantic_matching_v13.json` and `results/semantic_alignment_audit_v13.json`
  (run `run_20260907_234923_v13_373416cf`, section 6 "Public short-answer semantic experiment", cells 19-22; that run
  read the original Mohler release `ShortAnswerGrading_v2.0` with 87 question ids, and the LLM inside it was
  Qwen2.5-1.5B-Instruct, whereas the paper's LLM-channel numbers come from the v21 notebook with Qwen2.5-7B-Instruct
  on the 85-question cleaned corpus). The v13 notebook also contains earlier institutional and Qiyas sections; their
  cell outputs are aggregate metrics only and the underlying data are not included.

### Exp 4: multimodal fusion (synthetic cohort)

```
cd exp4_fusion
python exp4_fusion_generator.py --out results_dir     # regenerates exp4_cohort.csv, exp4_results.json, exp4_results.md
python make_fig6_ieee_style.py                        # Fig. 6
```

The cohort (1,500 learners, seeds 2026 / 42 / 7) is generated from the three-stage latent model documented in the
script docstring; no real learner data is involved.

## Methodology

The package reproduces the four experiments of the paper (Section 4, Materials and Methods, and Section 5, Results and Discussion):

* **Exp 1-A** (`exp1a_uci_federated/`): Algorithm 1 (Fed-Hist) on the public UCI Student Performance data; random forest, XGBoost, ANN, LSTM, a centralised deep model and the same model trained with FedAvg over four school × subject clients (25 rounds × 4 local epochs); accuracy, macro-F1 and ROC-AUC over five stratified 80/20 splits.
* **Exp 1-B** (`exp1b_institutional/`): Algorithm 1 on the institutional cohort (59,696 students, 12 colleges as federated clients) with Flower and Opacus: nine arms (centralised baseline, FedAvg, FedAvg + SecAgg+, DP-SGD at ε = 3 and 8, DP + SecAgg+, and two control arms), five seeds, common threshold policy (validation recall ≥ 80%), per-college recall with Wilson intervals, corrected TOST equivalence test, offline LiRA membership-inference attack, composition of the privacy budget, and the schedule-length sweep (5 to 50 rounds). The data are confidential; the notebooks document the protocol and keep their stored outputs.
* **Exp 2** (`exp2_hybrid_emo/`): Algorithm 2 (Hybrid-Emo) on CK+48 and FER2013; protocol described in the paper, code available on request.
* **Exp 3** (`exp3_semantic_knowledge/`): Algorithm 3 (Know-Evo) on the Mohler corpus: text-mining channel (six similarity features with ridge calibration), language-model channel (Qwen2.5-7B-Instruct, 4-bit NF4, greedy decoding) and their mean, evaluated against the mean of two human graders (RMSE, MAE, Pearson, Spearman, accuracy within ±0.5 and ±1) over five 70/30 splits.
* **Exp 4** (`exp4_fusion/`): the multimodal fusion engine on a generated cohort (N = 1,500, seeded) comparing single-modality predictors with early, weighted-late and attention-style fusion by ROC-AUC over five stratified folds.

All notebooks keep their executed outputs as the evidence behind the paper tables; the tables in `results/*/paper_tables_*.md` are generated by the notebooks from the result files.

## Changes in v1.1 (6 November 2026)

* All notebook comments, markdown cells and generated table headers are now in English (v1.0 contained Arabic working notes); numbers and code logic are unchanged.
* README expanded with the methodology summary and the Zenodo DOI; `CITATION.cff` updated.

## Data statement

| Data | Availability |
|---|---|
| UCI Student Performance (Exp 1-A) | Public. P. Cortez and A. Silva, "Using data mining to predict secondary school student performance", 2008. UCI Machine Learning Repository, dataset id 320, https://archive.ics.uci.edu/dataset/320/student+performance. Downloaded by the notebook; not redistributed here. |
| FER2013 (Exp 2) | Public (Kaggle, "Challenges in Representation Learning: Facial Expression Recognition Challenge"). Not redistributed here. |
| Mohler short-answer grading corpus (Exp 3) | Public. Mohler, Bunescu and Mihalcea (2011). The cleaned Hugging Face copy `nkazi/MohlerASAG` is loaded by the notebook; not redistributed here. |
| Institutional student cohort (Exp 1-B) | **Not shareable.** Student-level academic records of the Islamic University of Madinah are confidential. Access requests should be addressed to the Data Management Office of the Islamic University of Madinah. This package contains only aggregate result files (metrics, audits, manifests); no student records, identifiers or model weights are included. |
| Synthetic cohort (Exp 4) | Fully synthetic; the generator and the generated cohort are included. |

## Software versions

Recorded in `exp1b_institutional/results/run_v16/run_manifest_v16.json`: numpy 2.1.3, pandas 2.2.3, scipy 1.16.3,
scikit-learn 1.6.1, torch 2.11.0 (CPU), flwr 1.22.0, opacus 1.5.4. Exp 3 additionally uses transformers >= 4.45,
accelerate and bitsandbytes (Qwen2.5-7B-Instruct, 4-bit NF4, Tesla T4). Exp 1-A uses xgboost. Exp 2 requires
opencv-python, dlib and open_clip_torch (or `clip`). See `requirements.txt`.

## Integrity

`CHECKSUMS.sha256` lists the SHA-256 of every file. Verify with `sha256sum -c CHECKSUMS.sha256`.
Notebook outputs were kept; only Colab execution metadata (account identifiers) was removed.

## Licence

MIT Licence (see `LICENSE`). The generated Flower application in `exp1b_institutional/flower_app/` declares
`license = "Apache-2.0"` in its `pyproject.toml`, as generated by the notebook; the whole package is released under
MIT by the authors.

## Contributing

Questions, bug reports and suggestions are welcome through the GitHub issue tracker of this repository. Contributions that change the results (for example a re-run with other package versions) should be proposed as pull requests that keep the executed notebook outputs, update `CHECKSUMS.sha256` and describe the change in the pull request; the institutional data of Exp 1-B cannot be shared, so changes to that experiment can only be reviewed on the notebook logic and on the stored aggregate outputs. Please cite the paper and the Zenodo record when reusing the code.

## Citation

Alsubhy A., Alqurashi F., Abi Sen A., Mehmood R. (2026). FedEduAI: A Privacy-Aware Federated Multimodal Architecture
for Adaptive Learning Analytics and Decision Support. *PeerJ Computer Science* (under review).
Code and results: Zenodo, https://doi.org/10.5281/zenodo.23953300. See `CITATION.cff`.
