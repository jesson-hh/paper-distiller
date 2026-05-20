"""Local arxiv metadata mirror — SQLite + FTS5 + OAI-PMH sync.

Lets paper-distiller's search bypass arxiv's API rate limits by querying
a local copy of arxiv's metadata. Bootstrap from arxiv's published bulk
dump (~1.7M papers, ~1 GB compressed), keep current via OAI-PMH daily.
"""
