# API Basics — Notes

## What is an API?

An **API (Application Programming Interface)** is a set of rules that allows one piece of software to talk to another.

Think of it like a **waiter in a restaurant**:
- You (the client) place an order
- The waiter (the API) takes your request to the kitchen
- The kitchen (the server) prepares your food
- The waiter delivers the response back to you

---

## REST Architecture

**REST** = Representational State Transfer

REST APIs follow these principles:
1. **Client-Server** — Frontend and backend are separate
2. **Stateless** — Each request contains all info needed; no session stored on server
3. **Uniform Interface** — Standard HTTP methods (GET, POST, PUT, DELETE)
4. **Cacheable** — Responses can be cached

---

## JSON — The Language of APIs

Most REST APIs communicate using **JSON (JavaScript Object Notation)**:

```json
{
  "id": 1,
  "name": "Alice",
  "email": "alice@example.com",
  "age": 22
}
```

---

## URL Structure

```
https://api.example.com/users/42
         ↑               ↑    ↑
      base URL       resource  id
```

---

## What is Express.js?

**Express** is a minimal Node.js web framework for building REST APIs.

```js
const express = require('express');
const app = express();

app.get('/', (req, res) => {
  res.json({ message: 'Hello, World!' });
});

app.listen(3000);
```

---

## Testing APIs

Tools to test your API endpoints:
- **Postman** — GUI tool (download at postman.com)
- **Thunder Client** — VS Code extension
- **curl** — Command line: `curl http://localhost:3000/users`
