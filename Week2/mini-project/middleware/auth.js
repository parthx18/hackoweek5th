// ====================================================
// Auth Middleware — API Key Authentication
// ====================================================
// Protects routes by requiring a valid API key
// in the x-api-key request header.
//
// Usage: app.use('/api/notes', requireAuth, notesRouter)
// ====================================================

const VALID_API_KEY = 'hackoweek-2026-key';

const requireAuth = (req, res, next) => {
  const apiKey = req.headers['x-api-key'];

  if (!apiKey) {
    return res.status(401).json({
      error: 'Authentication Required',
      message: 'Please include an API key in the x-api-key header.',
      hint: `Use: x-api-key: ${VALID_API_KEY}`,
    });
  }

  if (apiKey !== VALID_API_KEY) {
    return res.status(403).json({
      error: 'Forbidden',
      message: 'Invalid API key provided.',
    });
  }

  next(); // Auth passed!
};

module.exports = requireAuth;
