# Deploying SmartWeatherAI to Google Cloud Run

Runs one prediction cycle per invocation, scheduled via Cloud Scheduler.

## Prerequisites
- Google Cloud account with billing enabled (free tier covers this workload)
- `gcloud` CLI installed and logged in: `gcloud auth login`
- Firebase project already exists (weather-app-2-920f0)
- `serviceAccountKey.json` in the repo root (used for Firebase Admin auth)

## 1. Set project
```bash
gcloud config set project weather-app-2-920f0
gcloud services enable run.googleapis.com cloudscheduler.googleapis.com artifactregistry.googleapis.com
```

## 2. Build & push the image
```bash
gcloud builds submit --tag gcr.io/weather-app-2-920f0/smart-weather-ai
```

## 3. Create the Cloud Run Job
```bash
gcloud run jobs create smart-weather-ai \
  --image gcr.io/weather-app-2-920f0/smart-weather-ai \
  --region asia-south1 \
  --memory 1Gi \
  --cpu 1 \
  --max-retries 1 \
  --task-timeout 15m
```

## 4. Schedule it every 30 min
```bash
gcloud scheduler jobs create http smart-weather-ai-schedule \
  --schedule "*/30 * * * *" \
  --location asia-south1 \
  --uri "https://asia-south1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/weather-app-2-920f0/jobs/smart-weather-ai:run" \
  --http-method POST \
  --oauth-service-account-email <YOUR-SA>@weather-app-2-920f0.iam.gserviceaccount.com
```

## Manual run (test)
```bash
gcloud run jobs execute smart-weather-ai --region asia-south1
```

## Local test (without deploying)
```bash
docker build -t smart-weather-ai .
docker run --rm smart-weather-ai
```

## Notes
- `serviceAccountKey.json` is baked into the image. For production, prefer Secret Manager and mount at runtime.
- Model + data files are re-generated inside the container on each run. To persist between runs, mount a GCS bucket via `--set-mounts`.
