# Week 2 — Build a REST API (Mini-Project)

## 🎯 Topic / Skill
Applying APIs & Backend Basics — **Building a Complete REST API**

## 📋 Mini-Project: Student Notes REST API

Build a fully functional REST API for managing student notes.  
This is a real-world backend project using Node.js + Express.

## 🗂️ Folder Structure
```
Week2/
├── README.md                ← You are here
└── mini-project/
    ├── package.json
    ├── server.js            ← Main entry point
    ├── routes/
    │   ├── notes.js         ← Notes CRUD routes
    │   └── users.js         ← Users routes
    ├── middleware/
    │   ├── auth.js          ← API key auth middleware
    │   └── logger.js        ← Request logger middleware
    └── data/
        └── store.js         ← In-memory data store
```

## 🚀 Features
- ✅ User management (create, list users)
- ✅ Notes CRUD (create, read, update, delete notes)
- ✅ Filter notes by user or tag
- ✅ API key authentication middleware
- ✅ Request logging middleware
- ✅ Proper error handling and status codes
- ✅ RESTful URL design

## 🔧 Setup & Run
```bash
cd mini-project
npm install
npm run dev
# Server runs on http://localhost:4000
```

## 📡 API Endpoints

### Users (Public — no auth needed)
| Method | Endpoint        | Description                     |
|--------|-----------------|---------------------------------|
| GET    | /api/users      | List all users                  |
| POST   | /api/users      | Create a user                   |
| GET    | /api/users/:id  | Get a user + their notes        |

### Notes (🔑 Auth required: `x-api-key: hackoweek-2026-key`)
| Method | Endpoint               | Description                  |
|--------|------------------------|------------------------------|
| GET    | /api/notes             | List all notes               |
| GET    | /api/notes?userId=1    | Filter notes by user         |
| GET    | /api/notes?tag=api     | Filter notes by tag          |
| POST   | /api/notes             | Create a note                |
| GET    | /api/notes/:id         | Get a note by ID             |
| PUT    | /api/notes/:id         | Update a note                |
| DELETE | /api/notes/:id         | Delete a note                |

## ⏱️ Duration: 3 Hours
- Hour 1: Set up project structure + data store + users API
- Hour 2: Build notes CRUD + middleware
- Hour 3: Test all endpoints with Postman / Thunder Client
