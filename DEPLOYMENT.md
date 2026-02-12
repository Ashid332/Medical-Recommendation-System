# Deployment Guide

This document provides comprehensive deployment instructions for the Medical Recommendation System.

## Table of Contents

1. [Heroku Deployment](#heroku-deployment)
2. [Docker Deployment](#docker-deployment)
3. [AWS Deployment](#aws-deployment)
4. [Local Development](#local-development)
5. [Environment Variables](#environment-variables)

## Heroku Deployment

### Prerequisites

- Heroku CLI installed
- Heroku account (free tier available)
- Git configured

### Steps

1. **Install Heroku CLI**
   ```bash
   npm install -g heroku
   ```

2. **Login to Heroku**
   ```bash
   heroku login
   ```

3. **Create Heroku App**
   ```bash
   heroku create medical-recommendation-system
   ```

4. **Set Environment Variables**
   ```bash
   heroku config:set FLASK_ENV=production
   heroku config:set APP_SECRET_KEY=your-secret-key-here
   ```

5. **Deploy Code**
   ```bash
   git push heroku main
   ```

6. **View Logs**
   ```bash
   heroku logs --tail
   ```

### Health Check

After deployment, verify the API is running:

```bash
curl https://medical-recommendation-system.herokuapp.com/api/health
```

## Docker Deployment

### Build Docker Image

```bash
docker build -t medical-recommendation:latest .
```

### Run Container

```bash
docker run -p 5000:5000 --env-file .env medical-recommendation:latest
```

## AWS Deployment

### Using Elastic Beanstalk

1. Install EB CLI
   ```bash
   pip install awsebcli
   ```

2. Initialize EB
   ```bash
   eb init -p python-3.10 medical-recommendation
   ```

3. Create Environment
   ```bash
   eb create production-env
   ```

4. Deploy
   ```bash
   eb deploy
   ```

## Local Development

### Setup

1. **Clone Repository**
   ```bash
   git clone https://github.com/Ashid332/Medical-Recommendation-System.git
   cd Medical-Recommendation-System
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Environment Variables**
   ```bash
   cp .env.example .env
   ```

5. **Run Application**
   ```bash
   python app.py
   ```

## Environment Variables

See `.env.example` for all required and optional environment variables.

### Critical Variables

- `FLASK_ENV`: production or development
- `APP_SECRET_KEY`: Secret key for Flask sessions
- `MODEL_PATH`: Path to trained model
- `SCALER_PATH`: Path to feature scaler

## Monitoring

After deployment, monitor:

- Application health endpoint: `/api/health`
- Application info: `/api/info`
- Model predictions: `/api/predict` (POST)

## Support

For deployment issues, contact: ashidulislam332@gmail.com
LinkedIn: linkedin.com/in/ashidulislam/
