# Docker Deployment Guide for SolarX

This guide provides comprehensive instructions for building, running, and deploying the SolarX web application using Docker.

## 📋 Prerequisites

- **Docker**: Version 20.10 or later
- **Docker Compose**: Version 2.0 or later (optional, for local development)
- **Git**: For cloning the repository

## 🏗️ Architecture Overview

The SolarX application uses a **multi-stage Docker build**:

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Frontend Build (node:18-alpine)                  │
│  - Install npm dependencies                                 │
│  - Build React app with Vite                                │
│  - Output: Optimized static files (~2-3 MB)                 │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Production Runtime (python:3.9-slim)             │
│  - Install Python dependencies                              │
│  - Copy SolarX modules and backend code                     │
│  - Copy frontend build from Stage 1                         │
│  - Serve both API and static files on port 8000             │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**
- Single container serves both frontend (React) and backend (FastAPI)
- Production-optimized build (~800MB - 1.2GB)
- Health checks for orchestration
- SPA routing support

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone <your-repo-url>
cd SolarX

# Build and run with docker-compose
docker-compose up

# Access the application
open http://localhost:8000
```

To stop:
```bash
docker-compose down
```

### Option 2: Docker CLI

```bash
# Build the Docker image
docker build -t solarx-web .

# Run the container
docker run -p 8000:8000 solarx-web

# Access the application
open http://localhost:8000
```

## 🧪 Testing the Deployment

After starting the container, verify it's working:

```bash
# Check health endpoint
curl http://localhost:8000/api/health

# Expected response:
# {"status":"healthy","model_loaded":true,"data_loaded":true}

# Check API documentation
open http://localhost:8000/docs

# Test frontend routing
open http://localhost:8000/demo
```

## 🌐 Accessing the Application

Once running, the following endpoints are available:

- **Web UI**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health
- **Interactive Demo**: http://localhost:8000/demo
- **Results Dashboard**: http://localhost:8000/results

## 🔧 Development Workflow

### With Docker (Hot Reload)

The `docker-compose.yml` includes volume mounts for development:

```bash
docker-compose up
```

Changes to backend code will require container restart:
```bash
docker-compose restart
```

### Without Docker (Faster for active development)

**Terminal 1 - Backend:**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm run dev
# Frontend runs on http://localhost:5173
```

## 🏭 Production Deployment

### Environment Variables

Configure these environment variables for production:

```bash
# Optional: Override port (some platforms require PORT env var)
PORT=8000

# Python unbuffered output (recommended for logging)
PYTHONUNBUFFERED=1
```

### Google Cloud Run

```bash
# 1. Build and submit to Google Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/solarx-web

# 2. Deploy to Cloud Run
gcloud run deploy solarx \
  --image gcr.io/YOUR_PROJECT_ID/solarx-web \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 1 \
  --port 8000

# 3. Get the deployed URL
gcloud run services describe solarx --region us-central1 --format 'value(status.url)'
```

**Required GCP Setup:**
1. Enable Cloud Run API
2. Enable Container Registry API
3. Set up billing
4. Configure `gcloud` CLI with your project

### Fly.io

```bash
# 1. Install flyctl
curl -L https://fly.io/install.sh | sh

# 2. Login to Fly.io
fly auth login

# 3. Launch app (interactive setup)
fly launch

# 4. Deploy
fly deploy

# 5. Open in browser
fly open
```

### Heroku

```bash
# 1. Login to Heroku
heroku login

# 2. Create app
heroku create your-app-name

# 3. Push container
heroku container:push web -a your-app-name

# 4. Release
heroku container:release web -a your-app-name

# 5. Open in browser
heroku open -a your-app-name
```

### GitHub Container Registry (GHCR)

The included GitHub Actions workflow automatically builds and pushes to GHCR:

```yaml
# Image will be available at:
ghcr.io/<your-username>/solarx:latest
```

To pull and run:
```bash
docker pull ghcr.io/<your-username>/solarx:latest
docker run -p 8000:8000 ghcr.io/<your-username>/solarx:latest
```

## 📊 Monitoring and Logs

### View logs in Docker

```bash
# Follow logs
docker-compose logs -f

# View specific container logs
docker logs <container-id> -f
```

### Health Checks

The container includes built-in health checks:

```bash
# Check container health status
docker ps

# Manually test health endpoint
docker exec <container-id> curl http://localhost:8000/api/health
```

## 🐛 Troubleshooting

### Issue: Container exits immediately

**Check logs:**
```bash
docker logs <container-id>
```

**Common causes:**
- Missing LSTM model file (`SolarX/src/lstm_solar_model.pth`)
- Missing data files in `SolarX/data/`
- Python dependency installation failure

### Issue: Frontend shows 404 on refresh

**Cause:** Static file serving not configured correctly

**Fix:** Ensure `StaticFiles` in `backend/app/main.py` has `html=True`:
```python
app.mount("/", StaticFiles(directory=frontend_dist, html=True), name="static")
```

### Issue: API calls fail from frontend

**Check CORS settings** in `backend/app/config.py`:
```python
cors_origins = ["*"]  # Allow all origins (adjust for production)
```

### Issue: Large image size (>2GB)

**Causes:**
- `.dockerignore` not excluding `node_modules/`
- Not using multi-stage build
- Using full `python:3.9` instead of `python:3.9-slim`

**Fix:**
1. Verify `.dockerignore` includes `node_modules/`
2. Check Dockerfile uses multi-stage build
3. Clean Docker cache: `docker system prune -a`

### Issue: Slow build times

**Enable BuildKit:**
```bash
export DOCKER_BUILDKIT=1
docker build -t solarx-web .
```

**Use layer caching:**
```bash
docker build --cache-from solarx-web:latest -t solarx-web .
```

## 🔐 Security Considerations

### Production Checklist

- [ ] Remove development volume mounts from `docker-compose.yml`
- [ ] Set specific CORS origins (not `["*"]`)
- [ ] Use secrets management for sensitive data (avoid baking into image)
- [ ] Run container as non-root user
- [ ] Keep base images updated (`python:3.9-slim`, `node:18-alpine`)
- [ ] Scan images for vulnerabilities: `docker scan solarx-web`

### Hardening the Dockerfile

Add non-root user:
```dockerfile
# Create non-root user
RUN useradd -m -u 1000 solarx
USER solarx
```

## 📦 Image Size Optimization

Current image size: ~800MB - 1.2GB

**Breakdown:**
- Base Python image: ~120MB
- PyTorch + dependencies: ~600MB
- Application code: ~50MB
- Frontend build: ~3MB

**Further optimization:**
1. Use PyTorch CPU-only build (reduces by ~300MB)
2. Remove unnecessary Python packages
3. Use alpine base (requires compilation, may increase build time)

## 🔄 CI/CD with GitHub Actions

The repository includes a GitHub Actions workflow (`.github/workflows/deploy.yml`) that:

1. Builds Docker image on every push
2. Runs health check tests
3. Pushes to GitHub Container Registry (GHCR)
4. Optional: Deploys to Google Cloud Run

**Required secrets** (for Cloud Run deployment):
- `GCP_PROJECT_ID`: Your Google Cloud project ID
- `GCP_SA_KEY`: Service account key JSON

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/docker/)
- [Vite Production Build](https://vitejs.dev/guide/build.html)
- [Google Cloud Run](https://cloud.google.com/run/docs)
- [Fly.io Docs](https://fly.io/docs/)

## 🆘 Getting Help

If you encounter issues:

1. Check logs: `docker logs <container-id>`
2. Verify health: `curl http://localhost:8000/api/health`
3. Test API: `curl http://localhost:8000/api/vendors`
4. Check this guide's Troubleshooting section
5. Open an issue on GitHub

---

**Last Updated:** 2026-02-15
**Docker Version:** 20.10+
**Docker Compose Version:** 2.0+
