// ====================================================
// Exercise 2 — CRUD Routes with In-Memory Data
// Week 1: APIs & Backend Basics
// ====================================================
// 🎯 Goal: Build full CRUD operations for a /students resource
// 📖 Learn: Route params, request bodies, all HTTP methods
// 💡 Data is stored in memory (no database yet!)
// ====================================================

const express = require('express');
const app = express();
const PORT = 3001;

app.use(express.json());

// -----------------------------------------------
// In-memory "database"
// -----------------------------------------------
let students = [
  { id: 1, name: 'Alice', course: 'Computer Science', year: 3 },
  { id: 2, name: 'Bob', course: 'Information Technology', year: 2 },
  { id: 3, name: 'Charlie', course: 'Electronics', year: 3 },
];
let nextId = 4;

// -----------------------------------------------
// GET /students — Read all students
// -----------------------------------------------
app.get('/students', (req, res) => {
  // Optional filter by year: GET /students?year=3
  const { year } = req.query;
  if (year) {
    const filtered = students.filter(s => s.year === parseInt(year));
    return res.json({ count: filtered.length, students: filtered });
  }
  res.json({ count: students.length, students });
});

// -----------------------------------------------
// GET /students/:id — Read one student
// -----------------------------------------------
app.get('/students/:id', (req, res) => {
  const id = parseInt(req.params.id);
  const student = students.find(s => s.id === id);

  if (!student) {
    return res.status(404).json({ error: `Student with id ${id} not found.` });
  }

  res.json(student);
});

// -----------------------------------------------
// POST /students — Create a new student
// Body: { "name": "...", "course": "...", "year": ... }
// -----------------------------------------------
app.post('/students', (req, res) => {
  const { name, course, year } = req.body;

  // Validation
  if (!name || !course || !year) {
    return res.status(400).json({
      error: 'Missing required fields',
      required: ['name', 'course', 'year'],
    });
  }

  const newStudent = { id: nextId++, name, course, year };
  students.push(newStudent);

  res.status(201).json({
    message: 'Student created successfully!',
    student: newStudent,
  });
});

// -----------------------------------------------
// PUT /students/:id — Update a student (full update)
// Body: { "name": "...", "course": "...", "year": ... }
// -----------------------------------------------
app.put('/students/:id', (req, res) => {
  const id = parseInt(req.params.id);
  const index = students.findIndex(s => s.id === id);

  if (index === -1) {
    return res.status(404).json({ error: `Student with id ${id} not found.` });
  }

  const { name, course, year } = req.body;
  if (!name || !course || !year) {
    return res.status(400).json({
      error: 'Missing required fields for full update',
      required: ['name', 'course', 'year'],
    });
  }

  students[index] = { id, name, course, year };

  res.json({
    message: 'Student updated successfully!',
    student: students[index],
  });
});

// -----------------------------------------------
// DELETE /students/:id — Delete a student
// -----------------------------------------------
app.delete('/students/:id', (req, res) => {
  const id = parseInt(req.params.id);
  const index = students.findIndex(s => s.id === id);

  if (index === -1) {
    return res.status(404).json({ error: `Student with id ${id} not found.` });
  }

  const deleted = students.splice(index, 1)[0];

  res.json({
    message: 'Student deleted successfully!',
    deleted,
  });
});

// -----------------------------------------------
// 404 Handler
// -----------------------------------------------
app.use((req, res) => {
  res.status(404).json({ error: 'Route not found', path: req.originalUrl });
});

// -----------------------------------------------
// Start Server
// -----------------------------------------------
app.listen(PORT, () => {
  console.log(`\n✅ Exercise 2 server running at http://localhost:${PORT}`);
  console.log('📌 CRUD routes for /students:');
  console.log(`   GET    http://localhost:${PORT}/students`);
  console.log(`   GET    http://localhost:${PORT}/students/:id`);
  console.log(`   POST   http://localhost:${PORT}/students`);
  console.log(`   PUT    http://localhost:${PORT}/students/:id`);
  console.log(`   DELETE http://localhost:${PORT}/students/:id`);
  console.log('\n🔍 Tip: Use Postman or Thunder Client to test all routes!\n');
});
