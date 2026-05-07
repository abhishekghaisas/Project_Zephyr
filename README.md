# AIport

> AI-powered conversational interface for airport gate utilization analysis at Seattle-Tacoma International Airport
>
> App Link: https://nam12.safelinks.protection.outlook.com/?url=https%3A%2F%2Fendearing-abundance-production-5a45.up.railway.app%2F&data=05%7C02%7Cghaisas.a%40northeastern.edu%7C9aa69bb3864e43280b8f08deaab7358d%7Ca8eec281aaa34daeac9b9a398b9215e7%7C0%7C0%7C639135901207507521%7CUnknown%7CTWFpbGZsb3d8eyJFbXB0eU1hcGkiOnRydWUsIlYiOiIwLjAuMDAwMCIsIlAiOiJXaW4zMiIsIkFOIjoiTWFpbCIsIldUIjoyfQ%3D%3D%7C0%7C%7C%7C&sdata=gV0GO941tjx0F6%2FuOto0JNdZ6hyuXQ0LBCs6KqSy5tQ%3D&reserved=0
>
> Username: admin
> Password: admin123

## Overview

This system converts natural language queries into SQL for gate utilization analytics, reducing report generation time from **2 weeks to under 2 minutes**.

**Client:** Port of Seattle  
**Institution:** Northeastern University Capstone Project

### Problem
- Manual BI reporting takes 1-2 weeks
- AerobahnDW contains 600+ unclear columns
- 25+ weekly requests creating backlog

### Solution
ChatGPT-like desktop interface using Google Gemini API for NLP-to-SQL conversion.

## Performance Metrics

| Metric | Result | Target |
|--------|--------|--------|
| Query Accuracy | 91% | 80% ✓ |
| Response Time | <45 sec | <120 sec ✓ |
| Time Reduction | 99.8% | — |
| Supported Query Types | 11 | 10+ ✓ |

## Tech Stack

| Layer | Technology |
|-------|------------|
| Desktop App | Electron |
| Frontend | React 18 + TypeScript + Vite |
| Backend | FastAPI + Python 3.8+ |
| AI/NLP | CodeLlama-7B + Claude 4.5 Haiku validation |
| Database | MySQL 8.0 |
| UI Components | shadcn/ui + Tailwind CSS |

## Screenshot

Sample Query and Response

<img width="1470" height="956" alt="Screenshot 2026-03-17 at 8 53 05 PM" src="https://github.com/user-attachments/assets/42c8e405-603d-4547-9b63-85a545a44a8c" />


## Quick Start

```bash
# 1. Clone repository
cd AIport

# 2. Setup database
mysql -u root -p < database/data/AIplane.sql

# 3. Configure backend
cd backend
cp .env.example .env
# Edit .env: add CLAUDE_API_KEY and DATABASE_PASSWORD
       .env: add MODAL_API_KEY for fine-tuned LLM

# 4. Install dependencies
pip install -r requirements.txt
cd ../frontend && npm install
cd ../ && npm install

# 5. Run application
# Terminal 1: Start backend
cd backend && python app.py

# Terminal 2: Start desktop app
npm run dev
```

See [QUICK_START.md](QUICK_START.md) for detailed instructions.

## Project Structure

```
AIport/
├── electron/
│   └── main.js              # Electron main process
├── backend/
│   ├── app.py               # Flask API server
│   ├── requirements.txt     # Python dependencies
│   └── .env.example         # Environment template
├── frontend/
│   ├── src/
│   │   ├── api/             # API client
│   │   ├── components/      # React components
│   │   └── App.tsx          # Main component
│   └── package.json
├── database/
│   ├── data/
│   │   ├── AIplane.sql      # Database dump
│   │   └── field_mapping.xlsx
│   └── scripts/
│       ├── db_manager.py    # Database CLI tool
│       └── config.py        # Database configuration
├── package.json             # Electron configuration
├── QUICK_START.md
└── README.md
```

## Supported Queries

The system supports 11 core query types:

| # | Query Type | Example |
|---|------------|---------|
| 1 | Taxi-In Performance | "Compare taxi-in times by aircraft type" |
| 2 | Taxi-Out Performance | "View taxi-out times across different hours" |
| 3 | Movement Area Occupancy | "Monitor aircraft in movement area" |
| 4 | Runway Occupancy | "Calculate runway occupancy times" |
| 5 | Wheels-Up Delay | "Identify delay patterns" |
| 6 | Weight Class Comparison | "Compare taxi duration by weight class" |
| 7 | Taxiway Utilization | "See which taxiway segments are most used" |
| 8 | Landing to In-Block | "Analyze landing to in-block duration" |
| 9 | Runway Utilization | "Show runway utilization rate" |
| 10 | Peak Hour Analysis | "Predict peak hours based on historical data" |
| 11 | Taxi Time Breakdown | "Compare taxi-in vs taxi-out by aircraft type" |

## Database Schema

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  aircraft_type  │       │     flight      │       │  flight_event   │
├─────────────────┤       ├─────────────────┤       ├─────────────────┤
│ aircraft_type PK│◄──────│ aircraft_type FK│       │ id PK           │
│ weight_class    │       │ id PK           │       │ call_sign FK    │
│ wake_category   │       │ call_sign       │◄──────│ operation       │
│ wingspan_ft     │       │ flight_number   │       │ event_type      │
│ wingspan_m      │       │ operation       │       │ event_time      │
└─────────────────┘       │ origin_airport  │       │ event_source    │
     16 records           │ destination     │       │ location        │
                          └─────────────────┘       └─────────────────┘
                               725 records              7,995 records
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/query` | POST | Process natural language query |
| `/api/health` | GET | Health check |
| `/api/stats` | GET | Database statistics |

## Team

| Name | Role |
|------|------|
| Abhishek Ghaisas | AI/ML Lead |
| Nishant Kumar | Project Manager |
| Ishan Chaudhary | Backend and Data Lead |
| Zhiqi Zhang | Frontend and Data Lead |

## Acknowledgments

- **Port of Seattle** — Project sponsor and data provider
- **Northeastern University** — Capstone program

## License

- **Code:** MIT License
- **Data:** Proprietary to Port of Seattle
