// ====================================================
// In-Memory Data Store
// Week 2 Mini-Project: Student Notes REST API
// ====================================================
// This simulates a database using JavaScript objects.
// In a real project you'd replace this with MongoDB,
// PostgreSQL, or another database.
// ====================================================

let users = [
  {
    id: 1,
    name: 'Alice',
    email: 'alice@hackoweek.com',
    course: 'Computer Science',
    createdAt: new Date('2026-08-01').toISOString(),
  },
  {
    id: 2,
    name: 'Bob',
    email: 'bob@hackoweek.com',
    course: 'Information Technology',
    createdAt: new Date('2026-08-02').toISOString(),
  },
];

let notes = [
  {
    id: 1,
    userId: 1,
    title: 'What is a REST API?',
    content: 'REST stands for Representational State Transfer. It uses HTTP methods like GET, POST, PUT, DELETE.',
    tags: ['api', 'basics'],
    createdAt: new Date('2026-08-10').toISOString(),
    updatedAt: new Date('2026-08-10').toISOString(),
  },
  {
    id: 2,
    userId: 1,
    title: 'Express Middleware',
    content: 'Middleware functions have access to req, res, and next. They run between request and response.',
    tags: ['express', 'middleware'],
    createdAt: new Date('2026-08-11').toISOString(),
    updatedAt: new Date('2026-08-11').toISOString(),
  },
  {
    id: 3,
    userId: 2,
    title: 'Node.js Event Loop',
    content: 'Node.js is non-blocking and uses an event loop to handle async operations efficiently.',
    tags: ['nodejs', 'async'],
    createdAt: new Date('2026-08-12').toISOString(),
    updatedAt: new Date('2026-08-12').toISOString(),
  },
];

let userIdCounter = 3;
let noteIdCounter = 4;

module.exports = {
  // User accessors
  getUsers: () => users,
  getUserById: (id) => users.find(u => u.id === id),
  addUser: (data) => {
    const user = {
      id: userIdCounter++,
      ...data,
      createdAt: new Date().toISOString(),
    };
    users.push(user);
    return user;
  },

  // Note accessors
  getNotes: (filter = {}) => {
    let result = [...notes];
    if (filter.userId) {
      result = result.filter(n => n.userId === parseInt(filter.userId));
    }
    if (filter.tag) {
      result = result.filter(n => n.tags && n.tags.includes(filter.tag));
    }
    return result;
  },
  getNoteById: (id) => notes.find(n => n.id === id),
  addNote: (data) => {
    const note = {
      id: noteIdCounter++,
      ...data,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    };
    notes.push(note);
    return note;
  },
  updateNote: (id, data) => {
    const index = notes.findIndex(n => n.id === id);
    if (index === -1) return null;
    notes[index] = { ...notes[index], ...data, updatedAt: new Date().toISOString() };
    return notes[index];
  },
  deleteNote: (id) => {
    const index = notes.findIndex(n => n.id === id);
    if (index === -1) return null;
    return notes.splice(index, 1)[0];
  },
};
