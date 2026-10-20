// ====================================================
// Users Router
// Week 2 Mini-Project: Student Notes REST API
// ====================================================

const express = require('express');
const router = express.Router();
const store = require('../data/store');

// GET /api/users — List all users
router.get('/', (req, res) => {
  const users = store.getUsers();
  res.json({
    count: users.length,
    users,
  });
});

// GET /api/users/:id — Get user by ID
router.get('/:id', (req, res) => {
  const id = parseInt(req.params.id);
  const user = store.getUserById(id);

  if (!user) {
    return res.status(404).json({ error: `User with id ${id} not found.` });
  }

  // Also include the user's notes
  const notes = store.getNotes({ userId: id });
  res.json({ ...user, notes });
});

// POST /api/users — Create a new user
router.post('/', (req, res) => {
  const { name, email, course } = req.body;

  if (!name || !email || !course) {
    return res.status(400).json({
      error: 'Missing required fields',
      required: ['name', 'email', 'course'],
    });
  }

  // Check for duplicate email
  const existing = store.getUsers().find(u => u.email === email);
  if (existing) {
    return res.status(409).json({
      error: 'Conflict',
      message: `A user with email "${email}" already exists.`,
    });
  }

  const user = store.addUser({ name, email, course });
  res.status(201).json({
    message: 'User created successfully!',
    user,
  });
});

module.exports = router;
