// ====================================================
// Student Notes REST API — Main Server
// Week 2 Mini-Project
// ====================================================
// 📡 API Base URL: http://localhost:4000/api
// 🔑 Auth Key: hackoweek-2026-key  (header: x-api-key)
// ====================================================

const express = require('express');
const cors = require('cors');

const logger = require('./middleware/logger');
const requireAuth = require('./middleware/auth');
const usersRouter = require('./routes/users');
const notesRouter = require('./routes/notes');

const app = express();
const PORT = 4000;

// -----------------------------------------------
// Global Middleware
// -----------------------------------------------
app.use(cors());
app.use(express.json());
app.use(logger); // Log every request

// -----------------------------------------------
// API Info Route (Public)
// -----------------------------------------------
app.get('/', (req, res) => {
  res.json({
    name: '📚 Student Notes REST API',
    version: '1.0.0',
    project: 'Hackoweek — Week 2 Mini-Project',
    base: `http://localhost:${PORT}/api`,
    auth: 'Required for /api/notes — use header: x-api-key: hackoweek-2026-key',
    endpoints: {
      users: {
        'GET /api/users': 'List all users',
        'GET /api/users/:id': 'Get user + their notes',
        'POST /api/users': 'Create a user {name, email, course}',
      },
      notes: {
        'GET /api/notes': 'List notes (filter: ?userId=1 or ?tag=api)',
        'GET /api/notes/:id': 'Get a note by ID',
        'POST /api/notes': 'Create a note {userId, title, content, tags?}',
        'PUT /api/notes/:id': 'Update a note',
        'DELETE /api/notes/:id': 'Delete a note',
      },
    },
  });
});

// -----------------------------------------------
// Route Mounting
// -----------------------------------------------
// Users — public (no auth needed)
app.use('/api/users', usersRouter);

// Notes — protected (API key required)
app.use('/api/notes', requireAuth, notesRouter);

// -----------------------------------------------
// Global Error Handler
// -----------------------------------------------
app.use((err, req, res, next) => {
  console.error('❌ Unhandled Error:', err.stack);
  res.status(500).json({
    error: 'Internal Server Error',
    message: err.message,
  });
});

// -----------------------------------------------
// 404 Handler
// -----------------------------------------------
app.use((req, res) => {
  res.status(404).json({
    error: 'Endpoint not found',
    path: req.originalUrl,
    hint: 'Visit http://localhost:4000/ for the API docs',
  });
});

// -----------------------------------------------
// Start Server
// -----------------------------------------------
app.listen(PORT, () => {
  console.log('\n╔══════════════════════════════════════════════╗');
  console.log('║   📚 Student Notes REST API — Running!       ║');
  console.log('╚══════════════════════════════════════════════╝');
  console.log(`\n🌐 Base URL:  http://localhost:${PORT}`);
  console.log(`📄 API Docs:  http://localhost:${PORT}/`);
  console.log('\n📡 Endpoints:');
  console.log(`   GET  http://localhost:${PORT}/api/users`);
  console.log(`   POST http://localhost:${PORT}/api/users`);
  console.log(`   GET  http://localhost:${PORT}/api/notes         [auth required]`);
  console.log(`   POST http://localhost:${PORT}/api/notes         [auth required]`);
  console.log('\n🔑 API Key:   x-api-key: hackoweek-2026-key');
  console.log('\n📋 Watching for requests...\n');
});
