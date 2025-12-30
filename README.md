# Voice-ID Registry API

Voice identity protection, licensing, and monetization platform.

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/voiceid)

## 🚀 Quick Deploy

### Deploy to Railway (Recommended)

1. Fork this repository
2. Go to [railway.app](https://railway.app)
3. Click "New Project" → "Deploy from GitHub repo"
4. Select this repository
5. Railway auto-detects and deploys!

Your API will be live at: `https://your-app.up.railway.app`

### Custom Domain

1. In Railway dashboard, go to your project
2. Click on your service → Settings → Domains
3. Add your custom domain (e.g., `api.voicevault.net`)
4. Add the CNAME record to your DNS

## 📚 API Documentation

Once deployed, access the interactive docs:

- **Swagger UI**: `https://your-domain.com/docs`
- **ReDoc**: `https://your-domain.com/redoc`

## 🔧 Endpoints Overview

### Users
- `POST /api/v1/users` - Create user (creator/platform)
- `GET /api/v1/users/{id}` - Get user details

### Enrollment
- `POST /api/v1/enrollment/sessions` - Start enrollment
- `POST /api/v1/enrollment/sessions/{id}/samples` - Submit voice sample
- `POST /api/v1/enrollment/sessions/{id}/complete` - Complete enrollment

### Synthesis
- `POST /api/v1/synthesis/licenses` - Create license
- `POST /api/v1/synthesis/synthesize` - Generate watermarked audio

### Detection
- `POST /api/v1/detection/analyze` - Analyze audio
- `POST /api/v1/detection/evidence` - Generate evidence bundle

### Demo
- `POST /api/v1/demo/seed` - Seed demo data
- `DELETE /api/v1/demo/reset` - Reset demo data

## 🏃 Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn main:app --reload

# Open docs
open http://localhost:8000/docs
```

## 📄 License

Proprietary - Gooverio Labs © 2025

## 📧 Contact

- Website: [voicevault.net](https://voicevault.net)
- Email: contact@gooveriolabs.com
