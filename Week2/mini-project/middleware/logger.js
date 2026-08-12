// ====================================================
// Logger Middleware
// ====================================================
// Logs every request with timestamp, method, URL,
// and response time.
// ====================================================

const logger = (req, res, next) => {
  const start = Date.now();
  const timestamp = new Date().toISOString();

  // When the response finishes, log the result
  res.on('finish', () => {
    const duration = Date.now() - start;
    const statusColor =
      res.statusCode >= 500 ? '\x1b[31m' :  // Red for 5xx
      res.statusCode >= 400 ? '\x1b[33m' :  // Yellow for 4xx
      res.statusCode >= 200 ? '\x1b[32m' :  // Green for 2xx
      '\x1b[0m';

    console.log(
      `\x1b[90m[${timestamp}]\x1b[0m ` +
      `\x1b[36m${req.method.padEnd(7)}\x1b[0m ` +
      `${req.originalUrl.padEnd(40)} ` +
      `${statusColor}${res.statusCode}\x1b[0m ` +
      `\x1b[90m${duration}ms\x1b[0m`
    );
  });

  next();
};

module.exports = logger;
