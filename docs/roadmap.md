# Angela Roadmap

This document records possible future capabilities. Items here are product ideas, not committed release dates.

## Document ingestion and research RAG

Allow users to upload documents, with an initial focus on academic papers, and make their contents available to Angela's retrieval system.

Planned capabilities:

- Parse PDF, DOCX, and plain-text files, with OCR support for scanned documents.
- Extract useful academic metadata such as title, authors, abstract, sections, references, and page numbers.
- Split documents into searchable passages and store their text, metadata, and embeddings in the user's local database.
- Answer questions with traceable citations back to the source document and page or section.
- Detect duplicate or updated documents and allow users to re-index or delete their data completely.

## Autonomous conversations

Allow Angela to initiate and continue conversations when useful instead of responding only after direct user input.

Planned capabilities:

- Start conversations from schedules, reminders, monitored events, or relevant changes in connected services.
- Continue an unfinished conversation when a follow-up becomes relevant.
- Provide user-controlled quiet hours, frequency limits, notification channels, and per-feature opt-in controls.
- Include clear pause and disable controls, usage limits, and an explanation of why each autonomous message was sent.

## Google Drive and OneDrive semantic file search

Allow users to describe a file or image in natural language and have Angela locate it in an authorized Google Drive or OneDrive folder.

Planned capabilities:

- Connect through user-owned OAuth credentials with least-privilege folder access where supported.
- Search filenames, file metadata, document contents, OCR text, and image descriptions.
- Maintain a local searchable index and synchronize changes without copying unnecessary cloud data.
- Return the matching file's name, location, preview, and provider link.
- Let users choose which folders are indexed, revoke access, and delete the local index.

## Guiding principles

- Keep indexes, embeddings, and extracted information local by default.
- Require explicit consent before monitoring or sending proactive messages.
- Make every connected account removable and every local index inspectable and deletable.
- Preserve source links and citations so retrieved information can be verified.
