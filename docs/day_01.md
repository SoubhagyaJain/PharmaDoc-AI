# Day 1: a bounded input boundary

The working implementation is in `src/pharmadoc_ai/ingestion.py`; its tests include the original 15 cases, the eight-byte read bound, real PDF routing, corrupt input, encryption, and page limits. To learn independently, implement the function in a scratch module before comparing it with the working version. Keep production code intact.

## First principles

A filename extension is a convention, a signature is a small piece of evidence, and decoding checks actual structure. A malicious or malformed file can have a valid signature. These checks form separate layers rather than interchangeable guarantees.

Example: a text file containing `%PDF-1.7` passes the cheap PDF signature precheck but fails the PDF decoder. The test suite deliberately demonstrates both outcomes.

## Challenge

Implement `detect_file_type(path: Path, max_bytes: int = 20 * 1024 * 1024)` returning `pdf`, `png`, or `jpeg`. Reject nonpositive limits, missing paths, directories, empty files, unsupported extensions, signature mismatches, and sizes above the inclusive limit. Read no more than eight content bytes and never change the input.

Use `Path.is_file`, `suffix.lower`, `stat().st_size`, and `open('rb')`. Translate filesystem errors to `InvalidDocumentError`; do not return arbitrary `None` values on failure.

Run:

```powershell
uv run --locked pytest tests/test_ingestion.py -q
uv run --locked ruff check src tests
```

## Debugging

| Symptom | Check |
|---|---|
| Import failure | Project root, `uv run`, and the `pharmadoc_ai` package name |
| Every signature fails | Binary mode and bytes-versus-string comparisons |
| Uppercase filenames fail | Normalize suffix before lookup |
| Exactly-limit file fails | Reject with `size > limit`, not `>=` |
| Large memory use | Read eight bytes rather than the whole file |
| Directory causes an OS traceback | Check regular-file status first |

The precheck uses constant content-read memory and a fixed number of filesystem operations. Whole-document decoding later has different memory and CPU costs; the 20 MiB upload and pixel limits bound those operations separately.

## Completion checkpoint

Explain why each validation layer exists, pass the tests, demonstrate the read bound, and describe one file that passes the precheck but fails decoding. Record the commands and your own explanation in the learning log.

Interview questions:

1. Why are an extension and an uploaded MIME type insufficient to validate an input?
2. How would you enforce an upload limit when the request has no Content-Length header?
