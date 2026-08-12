# Week 1 — APIs & Backend Basics (Node.js)

## 🎯 Topic / Skill
APIs & Backend Basics using **Node.js / Express**

## 📋 Learning Objectives
- Understand what an API is and how REST works
- Set up a Node.js + Express server from scratch
- Learn HTTP methods: GET, POST, PUT, DELETE
- Understand request/response cycle
- Use tools like Postman or Thunder Client to test APIs
- Learn about middleware, routing, and status codes

## 🗂️ Folder Structure
```
Week1/
├── README.md          ← You are here
├── notes/
│   ├── api-basics.md  ← Theory notes on APIs
│   └── rest-concepts.md
└── exercises/
    ├── ex1-hello-server/   ← Simple Express server
    ├── ex2-routes/         ← CRUD routes practice
    └── ex3-middleware/     ← Middleware exploration
```

## 📚 Key Concepts to Learn

### What is an API?
An **Application Programming Interface (API)** allows two software systems to communicate.
A **REST API** uses HTTP requests to perform CRUD operations.

### HTTP Methods
| Method | Action       | Example            |
|--------|--------------|--------------------|
| GET    | Read data    | GET /users         |
| POST   | Create data  | POST /users        |
| PUT    | Update data  | PUT /users/:id     |
| DELETE | Delete data  | DELETE /users/:id  |

### HTTP Status Codes
| Code | Meaning               |
|------|-----------------------|
| 200  | OK                    |
| 201  | Created               |
| 400  | Bad Request           |
| 404  | Not Found             |
| 500  | Internal Server Error |

## 🔧 Setup
```bash
cd exercises/ex1-hello-server
npm install
node server.js
```
