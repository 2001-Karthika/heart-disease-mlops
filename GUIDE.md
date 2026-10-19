# Complete step-by-step guide (AIMLCZG523 – MLOps Assignment 1)

Follow the parts **in order**. Each part ends with a **📸 Screenshot** list – save every screenshot into the
`screenshots/` folder using the suggested file names; you need them for the report and the video.

Commands are given for **Linux/macOS (bash)** and, where different, **Windows (PowerShell)**.
Replace `YOUR_GITHUB_USERNAME` everywhere.

---

## Part 0 – Install the tools (once)

| Tool | Why | Check |
|---|---|---|
| Python **3.11** (3.10–3.12 also OK) | run code | `python --version` |
| Git | version control | `git --version` |
| Docker Desktop | containers (+ local Kubernetes) | `docker --version` |
| kubectl | talk to Kubernetes | `kubectl version --client` |
| Minikube *(skip if you use Docker Desktop's Kubernetes)* | local cluster | `minikube version` |
| A GitHub account | repo + Actions | |
| VS Code (optional) | editor | |

---

## Part 1 – Set up the project locally

```bash
unzip heart-disease-mlops.zip && cd heart-disease-mlops
python -m venv .venv
source .venv/bin/activate            # Windows PowerShell:  .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```
> PowerShell blocks activation? Run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**Project layout** (explain this in the video):
```
src/        data.py, features.py, eda.py, train.py, predict.py, api.py, logging_config.py, config.py
scripts/    download_data.py, load_test.py, smoke_test.sh
tests/      pytest unit tests (data, features, model, API)
notebooks/  01_eda, 02_model_training, 03_inference
k8s/        deployment.yaml, service.yaml, ingress.yaml, hpa.yaml, monitoring.yaml
monitoring/ prometheus.yml + Grafana provisioning/dashboard
.github/workflows/ci.yml     Dockerfile   docker-compose.yml
data/ models/ reports/ screenshots/
```

---

## Part 2 – Get the dataset (Task 1)

```bash
python scripts/download_data.py
```
This downloads the UCI Heart Disease data (id 45, Cleveland subset: 303 rows, 13 features + target),
writes `data/raw/heart_disease_raw.csv` and the cleaned file `data/processed/heart_clean.csv`.

**If the download fails** (firewall/lab network): open https://archive.ics.uci.edu/dataset/45/heart+disease,
click *Download*, unzip, copy `processed.cleveland.data` to `data/raw/`, then run the script again (it picks it up).

Cleaning decisions (put these in the report):
* `?` placeholders (in `ca`, `thal`) → NaN; all columns numeric.
* Target 0–4 → binary (0 = no disease, ≥1 = disease).
* Duplicates dropped.
* Missing values are **kept** in the cleaned CSV and **imputed inside the sklearn pipeline** (median / most-frequent) so
  imputation statistics are learned on training folds only → no data leakage.

---

## Part 3 – EDA (Task 1)

```bash
python -m src.eda                  # writes reports/figures/eda_*.png and reports/eda_summary.md
jupyter notebook notebooks/01_eda.ipynb    # Run All – nicer for screenshots
```
Figures produced: class balance, histograms (by target), categorical-vs-target counts, correlation heatmap,
boxplots, missing values.

📸 `screenshots/01_eda_class_balance.png`, `01_eda_histograms.png`, `01_eda_correlation.png`
(copy from `reports/figures/` or screenshot the notebook).

Write 5–6 observations for the report from **your real output** (class balance %, strongest correlations with target,
outliers in `chol`, which columns have missing values).

---

## Part 4 – Train, tune, track with MLflow (Tasks 2, 3, 4)

```bash
python -m src.train
```
What it does:
* stratified 80/20 split; preprocessing `ColumnTransformer` (impute+scale numeric, impute+one-hot categorical);
* **Logistic Regression** and **Random Forest**, each tuned with `GridSearchCV` (5-fold stratified, ROC-AUC);
* 5-fold cross-validation of the tuned models → accuracy, precision, recall, F1, ROC-AUC (mean ± std);
* hold-out test metrics + confusion matrix, ROC comparison, feature-importance plots;
* picks the winner by **CV ROC-AUC** (not by test score), saves `models/model.joblib` (full pipeline) and
  `models/model_metadata.json`;
* MLflow: one parent run + one child run per model with params, metrics, plots, the logged model and the cleaned dataset.

Open the MLflow UI (second terminal, same venv, project folder):
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```
→ http://localhost:5000 → experiment `heart-disease-classification`.

📸 `screenshots/02_mlflow_runs_list.png` (runs table), `02_mlflow_compare.png` (select both child runs → *Compare*),
`02_mlflow_run_artifacts.png` (one run's artifacts: plots/ + model/).

Fill the **model comparison table** in the report from `models/model_metadata.json` (or the MLflow compare view).
Reproducibility: seeds fixed (`RANDOM_STATE=42`), requirements files, whole preprocessing inside the saved pipeline.
Optionally freeze exact versions: `pip freeze > requirements.lock.txt`.

---

## Part 5 – Unit tests and lint (Task 5)

```bash
flake8 src tests scripts
pytest --cov=src
```
Tests cover cleaning, schema validation, preprocessing, training output, prediction contract and the API
(`/health`, `/predict` valid/invalid, `/metrics`). The tests use a small **synthetic** frame (tests/synthetic.py), so
they never need the internet.

📸 `screenshots/03_pytest_pass.png`, `03_flake8_clean.png`

---

## Part 6 – Run the API locally (Task 6, first half)

```bash
uvicorn src.api:app --port 8000
```
Open http://localhost:8000/docs (Swagger UI) → *POST /predict* → *Try it out*.

Command-line test (second terminal):
```bash
./scripts/smoke_test.sh
# or
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```
PowerShell:
```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict -ContentType "application/json" -Body '{"age":63,"sex":1,"cp":1,"trestbps":145,"chol":233,"fbs":1,"restecg":2,"thalach":150,"exang":0,"oldpeak":2.3,"slope":3,"ca":0,"thal":6}'
```
Expected response shape:
```json
{"prediction":1,"label":"Heart disease","confidence":0.87,"probability_disease":0.87,"model_name":"logistic_regression","model_version":"1.0.0"}
```
Logs go to the console and `logs/api.log` (one JSON line per request and per prediction).

📸 `screenshots/04_swagger_predict.png`, `04_curl_predict.png`

---

## Part 7 – Git repository, GitHub and CI/CD (Task 5)

1. On github.com → **New repository** → name `heart-disease-mlops`, **Public**, *no* README/gitignore/license (empty).
2. In the project folder:
```bash
git init
git branch -M main
git config user.name "Your Name"
git config user.email "you@example.com"
git add .
git status        # CHECK: data/, models/model.joblib, models/model_metadata.json, reports/figures/ are included; .venv, mlruns are not
git commit -m "Initial commit: end-to-end MLOps pipeline for heart disease prediction"
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/heart-disease-mlops.git
git push -u origin main
```
Authentication: GitHub no longer accepts passwords. Easiest: install GitHub CLI and run `gh auth login`, or create a
**Personal Access Token** (Settings → Developer settings → Tokens, scope `repo` + `workflow`) and paste it as the password.

3. Open the repo → **Actions** tab → workflow *CI/CD* runs automatically. Stages: flake8 → pytest (+coverage) →
   data prep → training (MLflow) → upload artifacts → Docker build + smoke test → push image to GHCR.
4. Click the run → scroll to **Artifacts** (`training-artifacts`: models, MLflow db, plots, junit/coverage reports).

📸 `screenshots/05_github_repo.png`, `05_actions_run_green.png`, `05_actions_job_steps.png`, `05_actions_artifacts.png`

**If a step fails:** open the failing step's log, fix locally, `git add . && git commit -m "fix: ..." && git push`.
Typical causes: flake8 line-length (>100), a dependency version problem, or data files not committed.

Later changes always follow the same loop: edit → `git add . && git commit -m "msg" && git push`.

---

## Part 8 – Docker (Task 6)

Make sure Docker Desktop is running.
```bash
docker build -t heart-disease-api:latest .
docker run -d --name heart-api -p 8000:8000 heart-disease-api:latest
docker ps
./scripts/smoke_test.sh                  # or the curl/Invoke-RestMethod command from Part 6
docker logs heart-api | tail
```
Stop/remove: `docker rm -f heart-api`.

📸 `screenshots/06_docker_build.png`, `06_docker_ps.png`, `06_docker_predict.png`

The image is slim (`requirements-api.txt`), runs as non-root, and has a `HEALTHCHECK`.
It needs `models/model.joblib` → that is why the model is committed (Part 7). The model must be trained with the same
scikit-learn version as the image (both resolve the latest ≥1.5 from the requirement files). If you ever see
*“InconsistentVersionWarning”* in `docker logs`, re-run `python -m src.train` in a fresh venv built from
`requirements-dev.txt`, commit `models/`, rebuild.

---

## Part 9 – Monitoring and logging (Task 8)

```bash
docker rm -f heart-api 2>/dev/null
docker compose up --build -d
python scripts/load_test.py --url http://localhost:8000 --n 300     # generates traffic (incl. a few invalid requests)
```
* API: http://localhost:8000/docs   • Raw metrics: http://localhost:8000/metrics
* Prometheus: http://localhost:9090 → Status → Targets (should be **UP**); try query `api_requests_total`
* Grafana: http://localhost:3000 (admin / admin) → Dashboards → **Heart Disease API**
  (request rate, p95 latency, predictions by class, error rate, average confidence, model-loaded).
* Request log: `tail -f logs/api.log` (or `docker compose logs -f api`).

📸 `screenshots/07_metrics_endpoint.png`, `07_prometheus_targets.png`, `07_grafana_dashboard.png`, `07_api_logs.png`

Stop: `docker compose down`.

---

## Part 10 – Kubernetes deployment (Task 7)

### Option A – Docker Desktop Kubernetes (simplest)
Docker Desktop → Settings → Kubernetes → *Enable Kubernetes* → wait for green.
```bash
kubectl config use-context docker-desktop
docker build -t heart-disease-api:latest .
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl get pods,svc
```
The service is a `LoadBalancer`; Docker Desktop exposes it at **http://localhost** (port 80).

### Option B – Minikube
```bash
minikube start --driver=docker
docker build -t heart-disease-api:latest .
minikube image load heart-disease-api:latest        # makes the local image available inside the cluster
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl rollout status deployment/heart-disease-api
kubectl get pods,svc
```
In a **separate terminal, keep it running**: `minikube tunnel` → the service gets an EXTERNAL-IP.
Alternative without tunnel: `minikube service heart-disease-api --url`.

### Verify
```bash
./scripts/smoke_test.sh http://<EXTERNAL-IP>          # or http://localhost for Docker Desktop
kubectl logs deployment/heart-disease-api --tail=20
kubectl describe deployment heart-disease-api | head -30
```
### Optional extras
```bash
minikube addons enable ingress && kubectl apply -f k8s/ingress.yaml     # host heart.local
kubectl apply -f k8s/monitoring.yaml                                    # in-cluster Prometheus + Grafana
kubectl port-forward svc/grafana 3000:3000
```
### Using the image from GitHub instead of building locally
After Part 7 the CI pushes `ghcr.io/YOUR_GITHUB_USERNAME/heart-disease-mlops:latest`. On GitHub → your profile →
Packages → the package → *Package settings* → **Change visibility → Public**. Then edit `k8s/deployment.yaml`:
`image: ghcr.io/YOUR_GITHUB_USERNAME/heart-disease-mlops:latest` and `imagePullPolicy: Always`.
(Docker: `docker pull ghcr.io/YOUR_GITHUB_USERNAME/heart-disease-mlops:latest`.)

### Public cloud (only if you want to – local Kubernetes is explicitly allowed)
GKE: push the image to Artifact Registry/GHCR, `gcloud container clusters create-auto heart-cluster --region <r>`,
`gcloud container clusters get-credentials ...`, the same `kubectl apply`. Delete the cluster afterwards to avoid charges.

📸 `screenshots/08_kubectl_get_pods_svc.png`, `08_k8s_predict_response.png`, `08_kubectl_logs.png`
(+ optional `08_minikube_dashboard.png` via `minikube dashboard`)

Clean up: `kubectl delete -f k8s/deployment.yaml -f k8s/service.yaml`, `minikube stop`.

---

## Part 11 – Pull from GitHub in the BITS Prayogshala lab and run

Lab machines differ (shared VM vs. browser lab). Check first: `git --version`, `python3 --version`, `docker --version`.
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/heart-disease-mlops.git
cd heart-disease-mlops
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                          # tests pass
python -m src.train                # reproduces the model and MLflow runs
uvicorn src.api:app --host 0.0.0.0 --port 8000      # run API (no Docker needed)
# if Docker exists:
docker build -t heart-disease-api . && docker run -d -p 8000:8000 --name heart-api heart-disease-api
# or just pull the CI-built image (repo/package must be public):
docker run -d -p 8000:8000 ghcr.io/YOUR_GITHUB_USERNAME/heart-disease-mlops:latest
./scripts/smoke_test.sh
```
No Docker/Kubernetes in the lab? Show the API running with uvicorn plus your recorded Docker/K8s screenshots, and say so
in the report. If pip is blocked, ask the lab admin for a PyPI mirror (`pip install -i <mirror-url> ...`).

📸 `screenshots/09_lab_clone_and_run.png` (show hostname/date + git clone + prediction)

---

## Part 12 – The video (≈ 6–9 minutes)

Record the screen with OBS Studio / Xbox Game Bar (Win+G) / QuickTime. Narrate. Suggested script:

| Time | Show | Say |
|---|---|---|
| 0:00 | Title slide / repo page | Problem, dataset, what you built |
| 0:30 | VS Code project tree | Folder structure, one line per folder |
| 1:15 | EDA notebook or figures | Class balance, correlations, cleaning decisions |
| 2:00 | `python -m src.train` output + MLflow UI compare | Two models, tuning, CV metrics, why the winner |
| 3:15 | `pytest` + `flake8` | Unit tests |
| 3:45 | `git push` → GitHub Actions run (green) + artifacts | CI/CD stages |
| 4:45 | `docker build/run` + curl `/predict` | Containerization, JSON in / prediction + confidence out |
| 5:45 | `kubectl apply`, `get pods,svc`, curl via LoadBalancer | Deployment |
| 6:45 | Load test → Grafana + `logs/api.log` | Monitoring |
| 7:45 | Pull in the lab (`git clone` → run) | Reproducibility |
| 8:30 | Wrap-up | Learnings, limitations |

Tips: increase terminal font (Ctrl +), close notifications, do a dry run first, keep the cluster and Docker already
running, and pre-open the tabs you need. Upload to Google Drive/YouTube (unlisted) and put the link in the report.

---

## Part 13 – The report (10 pages, Word/PDF)

Start from `reports/REPORT_TEMPLATE.md` – every section the assignment asks for is there, with prompts for what to write
and where each screenshot goes. Copy it into Word/Google Docs (or use `pandoc reports/REPORT.md -o report.pdf`),
paste **your real numbers** from `models/model_metadata.json`/MLflow, insert screenshots, export to PDF.
The architecture diagram is `reports/figures/architecture.png`.

---

## Part 14 – Final checklist mapped to the marking scheme

| Task (marks) | Evidence in repo |
|---|---|
| 1 EDA (5) | `scripts/download_data.py`, `src/data.py`, `src/eda.py`, `notebooks/01_eda.ipynb`, `reports/figures/eda_*.png` |
| 2 Features & models (8) | `src/features.py`, `src/train.py`, CV table in report |
| 3 MLflow (5) | `mlflow.db`/`mlartifacts` (CI artifact), screenshots 02_* |
| 4 Packaging (7) | `models/model.joblib` (full pipeline), `requirements*.txt`, MLflow model artifact |
| 5 CI/CD & tests (8) | `tests/`, `.github/workflows/ci.yml`, Actions screenshots |
| 6 Docker (5) | `Dockerfile`, `/predict`, screenshots 06_* |
| 7 Deployment (7) | `k8s/*.yaml`, screenshots 08_* |
| 8 Monitoring (3) | `/metrics`, `docker-compose.yml`, `monitoring/`, `logs/api.log`, screenshots 07_* |
| 9 Report (2) | PDF/DOCX, repo link, video link |

Submit: repo URL, report (≥ 10 pages), video, and the API access instructions (README “Run it” section).

## Troubleshooting
* `ModuleNotFoundError: src` → run commands from the project root; use `python -m src.train`, not `python src/train.py`.
* Port 8000/5000/3000 busy → change the left side of the mapping, e.g. `-p 8080:8000`.
* `ImagePullBackOff` in Kubernetes → image not inside the cluster: `minikube image load heart-disease-api:latest`
  and keep `imagePullPolicy: IfNotPresent`.
* `EXTERNAL-IP <pending>` on Minikube → start `minikube tunnel` and leave it running.
* Docker build says `models/` missing → run training and commit `models/model.joblib`.
* Actions: *Permission denied* on GHCR push → repo Settings → Actions → General → Workflow permissions → *Read and write*.
