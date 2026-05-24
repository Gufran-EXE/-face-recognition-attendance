# 📋 Setup Guide - Face Recognition Attendance System

Complete guide for setting up the development environment.

## Prerequisites

- **Python 3.10+** - [Download](https://www.python.org/downloads/)
- **Node.js 16+** and npm - [Download](https://nodejs.org/)
- **PostgreSQL 12+** - [Download](https://www.postgresql.org/download/)
- **Redis** - [Download](https://redis.io/download)
- **Git** - [Download](https://git-scm.com/)

## Windows Setup

### 1. Install PostgreSQL

```bash
# Download and install from https://www.postgresql.org/download/windows/
# During installation:
# - Set password for postgres user
# - Remember the password and port (default: 5432)
```

### 2. Install Redis

```bash
# Option 1: Using Windows Subsystem for Linux (WSL2)
wsl
curl -fsSL https://packages.redis.io/gpg.sh | sudo gpg --dearmor -o /usr/share/keyrings/redis-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/redis-archive-keyring.gpg] https://packages.redis.io/deb $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/redis.list
sudo apt-get update
sudo apt-get install redis-server
redis-server

# Option 2: Using Docker
docker run -d -p 6379:6379 redis:latest
```

### 3. Clone Repository

```bash
git clone https://github.com/Gufran-EXE/-face-recognition-attendance.git
cd -face-recognition-attendance
```

### 4. Backend Setup

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Copy and configure .env
copy .env.example .env
# Edit .env with your settings

# Run migrations (after Phase 1)
alembic upgrade head

# Start backend
cd backend
uvicorn app:app --reload
```

### 5. Frontend Setup

```bash
# In a new terminal
cd frontend
npm install
npm run dev
```

---

## macOS Setup

### 1. Install PostgreSQL

```bash
# Using Homebrew
brew install postgresql@15
brew services start postgresql@15

# Create user and database
createuser -P attendance_user  # Enter password when prompted
createdb -O attendance_user attendance_db
```

### 2. Install Redis

```bash
# Using Homebrew
brew install redis
brew services start redis
```

### 3. Clone Repository

```bash
git clone https://github.com/Gufran-EXE/-face-recognition-attendance.git
cd -face-recognition-attendance
```

### 4. Backend Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt

# Copy and configure .env
cp .env.example .env
# Edit .env with your settings

# Run migrations (after Phase 1)
alembic upgrade head

# Start backend
cd backend
uvicorn app:app --reload
```

### 5. Frontend Setup

```bash
# In a new terminal
cd frontend
npm install
npm run dev
```

---

## Linux (Ubuntu/Debian) Setup

### 1. Install PostgreSQL

```bash
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib

# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create user and database
sudo -u postgres createuser -P attendance_user
sudo -u postgres createdb -O attendance_user attendance_db
```

### 2. Install Redis

```bash
sudo apt-get install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server
```

### 3. Install Python and Node.js

```bash
# Python (usually pre-installed)
sudo apt-get install python3.10 python3.10-venv python3-pip

# Node.js
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs
```

### 4. Clone Repository

```bash
git clone https://github.com/Gufran-EXE/-face-recognition-attendance.git
cd -face-recognition-attendance
```

### 5. Backend Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt

# Copy and configure .env
cp .env.example .env
# Edit .env with your settings

# Run migrations (after Phase 1)
alembic upgrade head

# Start backend
cd backend
uvicorn app:app --reload
```

### 6. Frontend Setup

```bash
# In a new terminal
cd frontend
npm install
npm run dev
```

---

## Docker Setup (All Platforms)

### 1. Install Docker

- **Windows/Mac:** [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Linux:** [Docker Engine](https://docs.docker.com/engine/install/)

### 2. Clone Repository

```bash
git clone https://github.com/Gufran-EXE/-face-recognition-attendance.git
cd -face-recognition-attendance
```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env if needed
```

### 4. Run with Docker Compose

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Remove everything
docker-compose down -v
```

Services will be available at:
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`
- PostgreSQL: `localhost:5432`
- Redis: `localhost:6379`

---

## Environment Configuration

Create `.env` file with your settings:

```env
# Application
APP_NAME=Face Recognition Attendance System
APP_ENV=development
DEBUG=True
SECRET_KEY=your-dev-secret-key

# Database
DATABASE_URL=postgresql://attendance_user:password@localhost:5432/attendance_db
# For development with SQLite:
# DATABASE_URL=sqlite:///./attendance.db

# Redis
REDIS_URL=redis://localhost:6379/0

# Face Recognition
CONFIDENCE_THRESHOLD=0.6
DUPLICATE_CHECK_WINDOW_MINUTES=5
FACE_ENCODING_MODEL=hog  # or cnn for GPU

# JWT
JWT_SECRET_KEY=your-jwt-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# API
API_HOST=0.0.0.0
API_PORT=8000
CORS_ORIGINS=["http://localhost:3000", "http://localhost:5173"]

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

---

## Verification

### Check Backend

```bash
# Backend should be running at http://localhost:8000
# Visit http://localhost:8000/docs for API documentation
curl http://localhost:8000/health
```

### Check Frontend

```bash
# Frontend should be running at http://localhost:5173
# Open http://localhost:5173 in your browser
```

### Check Database

```bash
# PostgreSQL
psql -h localhost -U attendance_user -d attendance_db

# Or using Python
python -c "from sqlalchemy import create_engine; engine = create_engine('postgresql://attendance_user:password@localhost:5432/attendance_db'); print('Connected!' if engine.execute('SELECT 1') else 'Failed')"
```

### Check Redis

```bash
redis-cli ping
# Should return: PONG
```

---

## Troubleshooting

### PostgreSQL Connection Error

```bash
# Check if PostgreSQL is running
# Windows
pg_isready -h localhost -p 5432

# macOS/Linux
sudo systemctl status postgresql
```

### Redis Connection Error

```bash
# Check if Redis is running
redis-cli ping
# Should return: PONG
```

### Port Already in Use

```bash
# Find and kill process using port 8000
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux
lsof -i :8000
kill -9 <PID>
```

### Virtual Environment Issues

```bash
# Deactivate current environment
deactivate

# Remove and recreate
rm -rf venv  # or rmdir /s venv on Windows
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements-dev.txt
```

### Module Not Found

```bash
# Ensure virtual environment is activated
which python  # Should show venv path

# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements-dev.txt
```

---

## Next Steps

After setup is complete:

1. ✅ Review [README.md](README.md)
2. ✅ Check [ROADMAP.md](ROADMAP.md)
3. ✅ Read [CONTRIBUTING.md](CONTRIBUTING.md)
4. ✅ Start Phase 1 development

---

**Status:** Setup Guide Complete ✅

**Need Help?** Open an issue on GitHub or check Troubleshooting section above.
