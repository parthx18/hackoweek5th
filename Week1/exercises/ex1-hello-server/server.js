// ====================================================
// Exercise 1 — Hello Express Server
// Week 1: APIs & Backend Basics
// ====================================================
// 🎯 Goal: Create your very first Express server
// 📖 Learn: How to set up a server, define routes,
//           send JSON responses
// ====================================================

const express = require('express');
const app = express();
const PORT = 3000;

// Middleware to parse JSON request bodies
app.use(express.json());

// -----------------------------------------------
// ROUTE 1: Home route
// GET http://localhost:3000/
// -----------------------------------------------
app.get('/', (req, res) => {
  res.json({
    message: '🚀 Hello, World! Your Express server is running!',
    week: 'Week 1 — APIs & Backend Basics',
    tip: 'Try visiting /about or /greet?name=YourName',
  });
});

// -----------------------------------------------
// ROUTE 2: About route
// GET http://localhost:3000/about
// -----------------------------------------------
app.get('/about', (req, res) => {
  res.json({
    project: 'Hackoweek REST API Learning',
    topic: 'APIs & Backend Basics (Node.js)',
    activity: 'Whiteboard Challenge',
    duration: '3 hours',
  });
});

// -----------------------------------------------
// ROUTE 3: Greeting with query param
// GET http://localhost:3000/greet?name=Alice
// -----------------------------------------------
app.get('/greet', (req, res) => {
  const name = req.query.name || 'Stranger';
  res.json({
    greeting: `Hello, ${name}! Welcome to backend development! 👋`,
  });
});

// -----------------------------------------------
// ROUTE 4: Echo — POST body back
// POST http://localhost:3000/echo
// Body: { "message": "any text" }
// -----------------------------------------------
app.post('/echo', (req, res) => {
  const { message } = req.body;
  if (!message) {
    return res.status(400).json({ error: 'Please provide a "message" in the request body.' });
  }
  res.json({
    you_said: message,
    echo: message.split('').reverse().join(''),
    length: message.length,
  });
});

// -----------------------------------------------
// 404 Handler — catches unknown routes
// -----------------------------------------------
app.use((req, res) => {
  res.status(404).json({
    error: 'Route not found',
    tried: req.originalUrl,
    hint: 'Available routes: GET /, GET /about, GET /greet, POST /echo',
  });
});

// -----------------------------------------------
// Start the server
// -----------------------------------------------
app.listen(PORT, () => {
  console.log(`\n✅ Server running at http://localhost:${PORT}`);
  console.log('📌 Available routes:');
  console.log(`   GET  http://localhost:${PORT}/`);
  console.log(`   GET  http://localhost:${PORT}/about`);
  console.log(`   GET  http://localhost:${PORT}/greet?name=Alice`);
  console.log(`   POST http://localhost:${PORT}/echo   (body: { "message": "..." })`);
  console.log('\n🛑 Press Ctrl+C to stop the server\n');
});
