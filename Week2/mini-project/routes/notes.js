// ====================================================
// Notes Router
// Week 2 Mini-Project: Student Notes REST API
// ====================================================

const express = require('express');
const router = express.Router();
const store = require('../data/store');

// GET /api/notes — List notes (with optional filters)
// Query params: ?userId=1  ?tag=api
router.get('/', (req, res) => {
  const { userId, tag } = req.query;
  const notes = store.getNotes({ userId, tag });
  res.json({
    count: notes.length,
    filters: { userId: userId || null, tag: tag || null },
    notes,
  });
});

// GET /api/notes/:id — Get a note by ID
router.get('/:id', (req, res) => {
  const id = parseInt(req.params.id);
  const note = store.getNoteById(id);

  if (!note) {
    return res.status(404).json({ error: `Note with id ${id} not found.` });
  }

  res.json(note);
});

// POST /api/notes — Create a new note
router.post('/', (req, res) => {
  const { userId, title, content, tags } = req.body;

  if (!userId || !title || !content) {
    return res.status(400).json({
      error: 'Missing required fields',
      required: ['userId', 'title', 'content'],
      optional: ['tags (array of strings)'],
    });
  }

  // Validate the user exists
  const user = store.getUserById(parseInt(userId));
  if (!user) {
    return res.status(404).json({ error: `User with id ${userId} not found.` });
  }

  const note = store.addNote({
    userId: parseInt(userId),
    title,
    content,
    tags: tags || [],
  });

  res.status(201).json({
    message: 'Note created successfully!',
    note,
  });
});

// PUT /api/notes/:id — Update a note
router.put('/:id', (req, res) => {
  const id = parseInt(req.params.id);
  const existing = store.getNoteById(id);

  if (!existing) {
    return res.status(404).json({ error: `Note with id ${id} not found.` });
  }

  const { title, content, tags } = req.body;
  if (!title && !content && !tags) {
    return res.status(400).json({
      error: 'Provide at least one field to update',
      updatable: ['title', 'content', 'tags'],
    });
  }

  const updated = store.updateNote(id, {
    title: title || existing.title,
    content: content || existing.content,
    tags: tags || existing.tags,
  });

  res.json({
    message: 'Note updated successfully!',
    note: updated,
  });
});

// DELETE /api/notes/:id — Delete a note
router.delete('/:id', (req, res) => {
  const id = parseInt(req.params.id);
  const deleted = store.deleteNote(id);

  if (!deleted) {
    return res.status(404).json({ error: `Note with id ${id} not found.` });
  }

  res.json({
    message: 'Note deleted successfully!',
    deleted,
  });
});

module.exports = router;
