# MLOps Assignment 1 – Heart Disease Risk Prediction
**Course:** AIMLCZG523 – Machine Learning Operations  **Name:** <your name>  **ID:** <your BITS id>
**Repository:** https://github.com/YOUR_GITHUB_USERNAME/heart-disease-mlops
**Video:** <link>  **Deployed API / access:** <URL or "http://localhost (Docker Desktop K8s), see README">

> How to use this template: replace every `<<…>>` with your own results. Target ≈ 10 pages with screenshots.
> Numbers must come from YOUR runs (`models/model_metadata.json`, MLflow UI, `reports/eda_summary.md`).

## 1. Introduction (≈ ½ page)
Problem statement (predict heart disease risk from patient data and serve it as a monitored cloud-ready API),
dataset (UCI Heart Disease, Cleveland, <<n>> patients, 13 features, binary target), and the tools used
(scikit-learn, MLflow, pytest, GitHub Actions, Docker, Kubernetes, Prometheus/Grafana).

## 2. Setup / installation instructions (≈ 1 page)
Prerequisites, then the commands from README “Quick start” and “Run it”. Include how to obtain the dataset
(`python scripts/download_data.py`, manual fallback).

## 3. Data acquisition and EDA (≈ 2 pages)
* Data description table (feature, meaning, type) – see `src/config.py → FEATURE_DESCRIPTIONS`.
* Cleaning: `?` → NaN (columns <<ca, thal>>, <<n>> rows), duplicates <<n>>, target binarised.
* Class balance: <<x% / y%>> → <<balanced/imbalanced; consequence for metrics>>.
* Figures: `eda_class_balance.png`, `eda_histograms.png`, `eda_correlation_heatmap.png` (+ boxplots/categorical).
* 4–6 observations: <<strongest correlated features, outliers, relationships>>.

## 4. Feature engineering and modelling (≈ 2 pages)
* Pipeline: median impute + StandardScaler (numeric), most-frequent impute + one-hot (categorical); why inside the
  pipeline (no leakage, reproducible, same transform at inference).
* Models: Logistic Regression, Random Forest. Hyper-parameter grids (copy from `src/train.py`).
* Evaluation protocol: stratified 80/20 split, 5-fold stratified CV on the training split, selection by CV ROC-AUC.
* Results table:

| Model | CV acc | CV prec | CV recall | CV F1 | CV ROC-AUC | Test acc | Test recall | Test ROC-AUC |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | <<>> | <<>> | <<>> | <<>> | <<>> | <<>> | <<>> | <<>> |
| Random Forest | <<>> | <<>> | <<>> | <<>> | <<>> | <<>> | <<>> | <<>> |

* Selected model and justification (mention recall's importance in a screening setting); ROC and confusion-matrix figures;
  feature importance / coefficients. Limitations: small dataset (<<n>>), single-centre data, tuned on the same CV folds.

## 5. Experiment tracking summary (≈ 1 page)
What is logged (params, metrics, plots, model, dataset). Screenshots: runs list, compare view, run artifacts.
Short table of the runs and what you learned from comparing them.

## 6. Model packaging and reproducibility (≈ ½ page)
`models/model.joblib` = full sklearn Pipeline (+ `model_metadata.json`), MLflow model artifact, `requirements*.txt`,
fixed seeds, Python version, Docker image pins the runtime.

## 7. CI/CD pipeline and testing (≈ 1½ pages)
Unit-test list (what each test file checks), coverage <<%>>. Workflow stages (lint → test → data → train → upload
artifacts → docker build + smoke test → GHCR). Screenshots: green Actions run, job steps, artifacts.

## 8. Containerisation (≈ ½ page)
Dockerfile choices (slim base, non-root, healthcheck), `/predict` request/response example, screenshot of
build/run/curl.

## 9. Production deployment (≈ 1 page)
Where deployed (<<Minikube / Docker Desktop / GKE>>), manifests (`k8s/`), 2 replicas + probes + resources,
LoadBalancer service, how verified. Screenshots: `kubectl get pods,svc`, prediction via the service, logs.

## 10. Monitoring and logging (≈ ½ page)
Request logging (`logs/api.log`, JSON lines), Prometheus metrics exposed, Grafana dashboard panels. Screenshots.

## 11. Architecture diagram
Insert `reports/figures/architecture.png` and describe the flow in 4–5 sentences.

## 12. Conclusion and future work (≈ ½ page)
Learnings; improvements: data/drift monitoring, model registry + promotion, canary deploy, Helm, authentication,
bigger datasets, calibrated probabilities.

## References
UCI Heart Disease dataset; scikit-learn, MLflow, FastAPI, Prometheus, Grafana, Kubernetes documentation.
