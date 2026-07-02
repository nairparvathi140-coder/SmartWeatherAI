# Deploying SmartWeatherAI to Google Cloud Run (24/7 service)

This runs the prediction/retrain loop continuously in the cloud so your PC no
longer needs to be on. It deploys as a **Cloud Run Service** with
`--min-instances=1` (always warm) and authenticates to Firebase using the
service's own identity — **no key file is baked into the image**.

---

## What you need first
- A Google account with billing enabled on project **weather-station-orpl**
  (Cloud Run has a generous free tier; a tiny always-on instance is a few
  ₹/day at most).
- The `gcloud` CLI installed. Check with `gcloud --version`.
  Install: https://cloud.google.com/sdk/docs/install

---

## Step 1 — Log in and select the project
```bash
gcloud auth login
gcloud config set project weather-station-orpl
```

## Step 2 — Enable the required services (one time)
```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com
```

## Step 3 — Give the runtime service account Firebase access
The pipeline reads/writes Realtime Database. Grant the default compute service
account the Firebase Admin role so Application Default Credentials work:
```bash
PROJECT_NUMBER=$(gcloud projects describe weather-station-orpl --format="value(projectNumber)")
gcloud projects add-iam-policy-binding weather-station-orpl \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/firebase.admin"
```
(Windows PowerShell: run the two lines separately, and use
`$PROJECT_NUMBER = gcloud projects describe weather-station-orpl --format="value(projectNumber)"`.)

## Step 4 — Deploy (build + ship in one command)
From inside `C:\Users\vssva\OneDrive\Desktop\SmartWeatherAI`:
```bash
gcloud run deploy smart-weather-ai \
  --source . \
  --region asia-south1 \
  --min-instances 1 \
  --max-instances 1 \
  --memory 1Gi \
  --cpu 1 \
  --no-cpu-throttling \
  --no-allow-unauthenticated \
  --set-env-vars FIREBASE_DATABASE_URL=https://weather-station-orpl-default-rtdb.asia-southeast1.firebasedatabase.app
```
- `--source .` lets Cloud Build build the Dockerfile for you (no local Docker needed).
- `--min-instances 1 --no-cpu-throttling` keeps the loop running between cycles.
- `--no-allow-unauthenticated` — the health endpoint isn't public; the loop
  doesn't need inbound traffic.

First deploy takes ~3–5 min. When it finishes it prints a Service URL.

## Step 5 — Confirm it's running
```bash
gcloud run services logs read smart-weather-ai --region asia-south1 --limit 50
```
You should see `SMART WEATHER AI — service starting`, then training + cycle
logs. The dashboard will start showing fresh data.

**Now you can turn off your PC** — the pipeline runs in the cloud.

---

## Everyday operations
- **Watch logs live:** `gcloud run services logs tail smart-weather-ai --region asia-south1`
- **Redeploy after code changes:** re-run the Step 4 command.
- **Pause it (stop billing):** `gcloud run services update smart-weather-ai --region asia-south1 --min-instances 0`
  (loop stops; set back to 1 to resume.)
- **Delete entirely:** `gcloud run services delete smart-weather-ai --region asia-south1`

## Notes
- State that must survive restarts (predictions to validate, history) already
  lives in Firebase, so an occasional container restart is harmless — it
  retrains once on boot and continues.
- Local runs still work exactly as before: `python main.py` uses
  `serviceAccountKey.json`; Cloud Run uses its service identity automatically.
