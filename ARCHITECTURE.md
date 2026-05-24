# 🏗️ System Architecture

Complete architecture documentation for the Face Recognition Attendance System.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     Client Layer                             │
├──────────────────┬──────────────────┬───────────────────────┤
│   Web Dashboard  │   Mobile App     │   Webcam Feed         │
│   (React)        │   (React Native) │   (OpenCV/Python)     │
└────────┬─────────┴────────┬─────────┴──────────────┬────────┘
         │                  │                    │
         └──────────────────┼────────────────────┘
                            │ HTTP/WebSocket
┌──────────────────────────────────────────────────────────────┐
│                   API Gateway Layer                          │
│                    (FastAPI)                                 │
├──────────────────────────────────────────────────────────────┤
│  ┌─────────────┬─────────────┬────────────┬──────────────┐  │
│  │  Auth API   │ Employee    │ Attendance │ Face API     │  │
│  │  Routes     │ Routes      │ Routes     │ Routes       │  │
│  └─────────────┴─────────────┴────────────┴──────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │           Middleware & Services Layer                  │  │
│  │  ┌──────────────┬──────────────┬───────────────────┐  │  │
│  │  │ Auth Service │ Face Service │ Attendance Service│  │  │
│  │  └──────────────┴──────────────┴───────────────────┘  │  │
│  └───────────────────────────────────────��─────────────────┘  │
└──────────────┬──────────────────┬──────────────┬──────────────┘
               │                  │              │
┌──────────────┴──────────────────┴──────────────┴──────────────┐
│                   Data Layer                                  │
├───────────────────┬──────────────┬────────────────────────────┤
│   PostgreSQL      │   Redis      │   File Storage             │
│   Database        │   Cache      │   (Face Encodings, Images) │
└───────────────────┴──────────────┴────────────────────────────┘
```

## Component Architecture

### 1. Frontend Architecture (React)

```
Frontend/
├── Pages/
│   ├── LoginPage              # Authentication
│   ├── DashboardPage          # Real-time monitoring
│   ├── EmployeesPage          # Employee management
│   ├── AttendancePage         # Attendance records
│   ├── ReportsPage            # Analytics
│   └── SettingsPage           # Configuration
│
├── Components/
│   ├── Common/                # Reusable components
│   │   ├── Header
│   │   ├── Sidebar
│   │   ├── Modal
│   │   └── LoadingSpinner
│   ├── Forms/
│   │   ├── LoginForm
│   │   ├── EmployeeForm
│   │   └── UploadFaceForm
│   └── Charts/
│       ├── AttendanceChart
│       └── StatCard
│
├── Services/
│   ├── api.js                 # API client with interceptors
│   ├── auth.service.js        # Auth operations
│   └── employee.service.js    # Employee operations
│
├── Store/
│   ├── authStore              # Auth state (Zustand)
│   ├── employeeStore          # Employee state
│   └── attendanceStore        # Attendance state
│
├── Hooks/
│   ├── useAuth                # Authentication hook
│   ├── useAttendance          # Attendance fetching
│   └── useNotification        # Toast/notification
│
└── Styles/
    ├── tailwind.css           # Tailwind configuration
    └── globals.css            # Global styles
```

### 2. Backend Architecture (FastAPI)

#### Route Layer
```python
# app.py - Main application
FastAPI()
├── /api/auth/              # Authentication endpoints
│   ├── POST login
│   ├── POST logout
│   ├── POST refresh_token
│   └── POST verify_token
├── /api/employees/         # Employee management
│   ├── GET (list)
│   ├── POST (create)
│   ├── GET {id}
│   ├── PUT {id}
│   └── DELETE {id}
├── /api/attendance/        # Attendance records
│   ├── GET (list)
│   ├── POST check-in
│   ├── GET {employee_id}
│   └── GET /report
├── /api/faces/             # Face management
│   ├── POST upload
│   ├── POST train
│   ├── GET {employee_id}
│   └── DELETE {id}
└── /api/health/            # Health check
    └── GET status
```

### 3. Database Schema

#### Employees Table
```sql
Table: employees
├── id (PK, Integer)
├── name (String, NOT NULL)
├── email (String, UNIQUE, NOT NULL)
├── department (String)
├── status (Enum: active, inactive)
├── created_at (Timestamp)
└── updated_at (Timestamp)
```

#### Face Encodings Table
```sql
Table: face_encodings
├── id (PK, Integer)
├── employee_id (FK → employees.id)
├── encoding (Binary, NOT NULL)
├── image_path (String)
├── created_at (Timestamp)
└── metadata (JSON)
```

#### Attendance Logs Table
```sql
Table: attendance_logs
├── id (PK, Integer)
├── employee_id (FK → employees.id)
├── check_in (Timestamp, NOT NULL)
├── check_out (Timestamp)
├── date (Date)
├── status (Enum: present, absent, late)
└── created_at (Timestamp)
```

## Security Architecture

```
┌─────────────────────────────────────┐
│     API Request                     │
└────────────┬────────────────────────┘
             │
             ▼
    ┌────────────────────┐
    │ CORS Middleware    │
    │ (Validate origin)  │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ Auth Middleware    │
    │ (Verify JWT)       │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ Authorization      │
    │ (Check role/perms) │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ Input Validation   │
    │ (Pydantic)         │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ Business Logic     │
    │ (Service layer)    │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ Database Access    │
    │ (ORM queries)      │
    └────────┬───────────┘
             │
             ▼
    ┌────────────────────┐
    │ Response           │
    │ (JSON)             │
    └────────────────────┘
```

---

**Last Updated:** May 24, 2025
