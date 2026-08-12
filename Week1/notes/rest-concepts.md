# REST Concepts — Notes

## CRUD Operations

Every REST API is built around **CRUD**:

| CRUD   | HTTP Method | SQL Equivalent |
|--------|-------------|----------------|
| Create | POST        | INSERT         |
| Read   | GET         | SELECT         |
| Update | PUT / PATCH | UPDATE         |
| Delete | DELETE      | DELETE         |

---

## RESTful Endpoint Design

### Resource Naming — Use Nouns, Not Verbs

✅ Good:
```
GET    /users
GET    /users/1
POST   /users
PUT    /users/1
DELETE /users/1
```

❌ Bad:
```
GET  /getUsers
POST /createUser
GET  /deleteUser?id=1
```

---

## Request Anatomy

```
POST /users HTTP/1.1
Host: localhost:3000
Content-Type: application/json

{
  "name": "Bob",
  "email": "bob@test.com"
}
```

Parts:
- **Method**: POST
- **Path**: /users
- **Headers**: Content-Type tells server it's JSON
- **Body**: The data being sent

---

## Response Anatomy

```json
HTTP/1.1 201 Created
Content-Type: application/json

{
  "id": 5,
  "name": "Bob",
  "email": "bob@test.com"
}
```

---

## Query Parameters vs Route Parameters

### Route Parameters — specific resource
```
GET /users/42       → get user with id 42
```

### Query Parameters — filtering/sorting
```
GET /users?role=admin&sort=name
```

In Express:
```js
// Route param
app.get('/users/:id', (req, res) => {
  console.log(req.params.id); // "42"
});

// Query param
app.get('/users', (req, res) => {
  console.log(req.query.role);  // "admin"
  console.log(req.query.sort);  // "name"
});
```

---

## Middleware

Middleware is a function that runs **between** the request and the response:

```
Request → [Middleware 1] → [Middleware 2] → Route Handler → Response
```

Common middleware:
- `express.json()` — Parse JSON body
- `cors()` — Allow cross-origin requests
- Custom logging, auth checks, etc.
