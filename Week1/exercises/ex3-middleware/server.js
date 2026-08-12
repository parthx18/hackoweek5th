// ====================================================
// Exercise 3 — Middleware Exploration
// Week 1: APIs & Backend Basics
// ====================================================
// 🎯 Goal: Understand how middleware works in Express
// 📖 Learn: Custom logging, request validation,
//           simple authentication middleware, CORS
// ====================================================

const express = require('express');
const cors = require('cors');
const app = express();
const PORT = 3002;

// -----------------------------------------------
// 1. Built-in Middleware
// -----------------------------------------------
app.use(express.json());       // Parse JSON bodies
app.use(cors());               // Allow cross-origin requests

// -----------------------------------------------
// 2. Custom Logging Middleware
//    Runs on EVERY request
// -----------------------------------------------
app.use((req, res, next) => {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] ${req.method} ${req.url}`);

  // Attach request time to the request object
  req.requestTime = timestamp;

  next(); // ← IMPORTANT: call next() to move to the next middleware
});

// -----------------------------------------------
// 3. Custom "API Key" Auth Middleware
//    Applied only to /protected routes
// -----------------------------------------------
const requireApiKey = (req, res, next) => {
  const apiKey = req.headers['x-api-key'];

  if (!apiKey || apiKey !== 'hackoweek-secret-key') {
    return res.status(401).json({
      error: 'Unauthorized',
      hint: 'Add header: x-api-key: hackoweek-secret-key',
    });
  }

  next(); // Key is valid — proceed
};

// -----------------------------------------------
// 4. Request Size Validator Middleware
// -----------------------------------------------
const validateBody = (requiredFields) => (req, res, next) => {
  const missing = requiredFields.filter(f => !(f in req.body));
  if (missing.length > 0) {
    return res.status(400).json({
      error: 'Missing required fields',
      missing,
    });
  }
  next();
};

// -----------------------------------------------
// ROUTES
// -----------------------------------------------

// Public route — no auth needed
app.get('/', (req, res) => {
  res.json({
    message: '🔧 Middleware Exercise Server',
    requestTime: req.requestTime,
    routes: {
      public: ['GET /', 'GET /public'],
      protected: ['GET /protected', 'POST /protected/data'],
    },
  });
});

app.get('/public', (req, res) => {
  res.json({
    message: '✅ This is a public endpoint — anyone can access it!',
    requestTime: req.requestTime,
  });
});

// Protected route — requires API key header
app.get('/protected', requireApiKey, (req, res) => {
  res.json({
    message: '🔐 You accessed a protected route! API key was valid.',
    requestTime: req.requestTime,
    secretData: '🎉 Hackoweek Week 1 is awesome!',
  });
});

// Protected route with body validation
app.post(
  '/protected/data',
  requireApiKey,
  validateBody(['title', 'content']),
  (req, res) => {
    res.status(201).json({
      message: '✅ Data received and validated!',
      data: req.body,
      receivedAt: req.requestTime,
    });
  }
);

// -----------------------------------------------
// Error-Handling Middleware (4 params!)
// -----------------------------------------------
app.use((err, req, res, next) => {
  console.error('❌ Error:', err.message);
  res.status(500).json({
    error: 'Internal Server Error',
    message: err.message,
  });
});

// 404 Handler
app.use((req, res) => {
  res.status(404).json({ error: 'Route not found', path: req.originalUrl });
});

// -----------------------------------------------
// Start Server
// -----------------------------------------------
app.listen(PORT, () => {
  console.log(`\n✅ Exercise 3 (Middleware) running at http://localhost:${PORT}`);
  console.log('\n📌 Try these:');
  console.log(`   GET  http://localhost:${PORT}/public`);
  console.log(`   GET  http://localhost:${PORT}/protected`);
  console.log('        → Add header: x-api-key: hackoweek-secret-key');
  console.log('\n💡 Watch this console for request logs!\n');
});
