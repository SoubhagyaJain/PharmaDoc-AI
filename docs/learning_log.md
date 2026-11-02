# Learning checkpoints

The code is implemented because full implementation was explicitly requested. Use the roadmap as a sequence of exercises, rather than trying to read every module at once.

For each session, record:

1. The concept in your own words and why the pipeline needs it.
2. A tiny example you can predict by hand.
3. Your attempted implementation or extension.
4. Test commands and actual output.
5. A failure you investigated and the evidence behind the fix.
6. CPU/memory/latency implications where relevant.
7. Answers to the two interview questions.

## Useful implementation lessons from this build

- Pydantic union branches need their own strict scalar types; applying a single `strict` constraint to a union is not supported by the installed version.
- Malformed native PDF decoding can retain Windows file handles through exception lifetimes. Opening already size-bounded memory streams prevents cleanup from depending on native decoder file-handle release.
- A standalone worker in an `ocr` directory can accidentally import its neighboring `paddle.py` instead of the native Paddle package. Remove the worker directory from its import search path before importing Paddle.
- Rotated OCR must establish reading order before evidence boxes are mapped back to the original page. Sorting restored coordinates can scramble otherwise correct recognition.
- Repeated strengths and doses must be labeled within their own medication blocks. A global first-substring search creates incorrect or missing training annotations.
- Checkbox borders, dots, and ticks have different geometries. A fill threshold alone can confidently classify a small dot as a checked box.

These are engineering findings, not reasons to assume unseen cases are solved. Keep the regression tests and inspect new failure categories honestly.
