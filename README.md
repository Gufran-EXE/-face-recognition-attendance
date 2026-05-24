# 🎯 Face Recognition Attendance System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-blue.svg)](https://react.dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A modern, production-ready **Face Recognition-based Attendance System** built with Python, FastAPI, React, and PostgreSQL. This system captures live video from your webcam, detects and recognizes registered faces, and automatically logs attendance with real-time dashboard monitoring.

## ✨ Features

### Core Features
- ✅ **Real-time Face Recognition** - Live webcam face detection and recognition
- ✅ **Automated Attendance Logging** - Automatic attendance records with timestamp
- ✅ **Employee Management** - Add, edit, and remove employees with multiple face images
- ✅ **Attendance Reports** - Daily, weekly, and monthly attendance summaries
- ✅ **Web Dashboard** - Real-time monitoring and attendance visualization
- ✅ **REST API** - Complete API for third-party integrations

### Advanced Features
- 🔐 **Authentication & Authorization** - JWT-based with role-based access control
- 📊 **Analytics & Reports** - Detailed attendance analytics and charts
- 🔔 **Notifications** - Email and SMS alerts for check-ins
- 📱 **Mobile Ready** - Responsive design for mobile and tablet devices
- 🗄️ **Database** - PostgreSQL with proper schema and migrations
- 🚀 **Scalable Architecture** - Microservices-ready with caching (Redis)

## 📋 Prerequisites

- **Python 3.10+**
- **PostgreSQL 12+** (or SQLite for development)
- **Node.js 16+** and npm
- **Redis** (for caching)
- **Webcam** (for face recognition)

## 🚀 Quick Start

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/Gufran-EXE/-face-recognition-attendance.git
cd -face-recognition-attendance

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements-dev.txt

# Copy environment file
cp .env.example .env
# Edit .env with your configuration

# Run database migrations
alembic upgrade head

# Start the backend server
cd backend
uvicorn app:app --reload
```

Backend will be available at `http://localhost:8000`

### Frontend Setup

```bash
# In a new terminal, navigate to frontend
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will be available at `http://localhost:5173`

## 📁 Project Structure

```
face-recognition-attendance/
├── backend/
│   ├── app.py                      # FastAPI application
│   ├── config.py                   # Configuration management
│   ├── routes/                     # API endpoints
│   │   ├── __init__.py
│   │   ├── auth.py                # Authentication routes
│   │   ├── employees.py           # Employee management
│   │   ├── attendance.py          # Attendance records
│   │   └── faces.py               # Face management
│   ├── models/
│   │   ├── __init__.py
│   │   ├── database.py            # SQLAlchemy models
│   │   └── schemas.py             # Pydantic schemas
│   ├── services/
│   │   ├── __init__.py
│   │   ├── face_service.py        # Face recognition logic
│   │   ├── attendance_service.py  # Attendance business logic
│   │   └── auth_service.py        # Authentication logic
│   ├── database/
│   │   ├── __init__.py
│   │   └── engine.py              # Database connection
│   ├── middleware/
│   │   ├── __init__.py
│   │   └── auth.py                # Authentication middleware
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py              # Logging configuration
│   │   └── exceptions.py          # Custom exceptions
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── pages/                 # Page components
│   │   ├── components/            # Reusable components
│   │   ├── services/              # API service layer
│   │   ├── hooks/                 # Custom React hooks
│   │   ├── store/                 # State management
│   │   ├── styles/                # Global styles
│   │   └── App.jsx
│   ├── public/
│   ├── package.json
│   └── vite.config.js
│
├── scripts/
│   ├── register_face.py           # Register new employee
│   ├── manage_employees.py        # Manage employees
│   └── train_encodings.py         # Pre-train face encodings
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # Pytest configuration
│   ├── test_face_service.py
│   ├── test_attendance_logger.py
│   └── test_auth.py
│
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── ROADMAP.md
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .env.example
└── README.md
```

## 🔧 Configuration

Copy `.env.example` to `.env` and configure:

```env
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/attendance_db

# Face Recognition
CONFIDENCE_THRESHOLD=0.6
DUPLICATE_CHECK_WINDOW_MINUTES=5

# JWT
JWT_SECRET_KEY=your-secret-key

# API
API_HOST=0.0.0.0
API_PORT=8000
```

## 📚 API Documentation

Once the backend is running, visit:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_face_service.py -v

# Run in watch mode
pytest-watch
```

## 🐳 Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

## 📊 Database Schema

### Employees Table
```sql
CREATE TABLE employees (
  id SERIAL PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  email VARCHAR(100) UNIQUE NOT NULL,
  department VARCHAR(50),
  status VARCHAR(20) DEFAULT 'active',
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Face Encodings Table
```sql
CREATE TABLE face_encodings (
  id SERIAL PRIMARY KEY,
  employee_id INTEGER NOT NULL REFERENCES employees(id),
  encoding BYTEA NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Attendance Logs Table
```sql
CREATE TABLE attendance_logs (
  id SERIAL PRIMARY KEY,
  employee_id INTEGER NOT NULL REFERENCES employees(id),
  check_in TIMESTAMP,
  check_out TIMESTAMP,
  date DATE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

## 🔐 Security

- JWT-based authentication
- Password hashing with bcrypt
- CORS protection
- Rate limiting
- SQL injection prevention with ORM
- Environment variable configuration
- Audit logging

## 📈 Performance

- Face encoding caching (Redis)
- Database query optimization
- Lazy loading for images
- Async/await for non-blocking operations
- Response compression

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and development process.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [face_recognition](https://github.com/ageitgey/face_recognition) - Face recognition library
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [React](https://react.dev/) - JavaScript library for UI
- [OpenCV](https://opencv.org/) - Computer vision library

## 📞 Support

For support, email: support@attendance-system.com or open an issue on GitHub.

## 🗺️ Roadmap

See [ROADMAP.md](ROADMAP.md) for detailed development roadmap and upcoming features.

---

**Status:** 🔄 In Development (Phase 0: Foundation)

**Last Updated:** May 24, 2025
