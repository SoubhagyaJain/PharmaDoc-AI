"""Public errors contain stable codes, never document contents."""


class DocumentError(ValueError):
    def __init__(self, message: str, code: str = "invalid_document", status_code: int = 422):
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class EngineUnavailable(DocumentError):
    def __init__(self, engine: str):
        super().__init__(
            f"The configured {engine} engine is unavailable.", "engine_unavailable", 503
        )


class ProcessingTimeout(DocumentError):
    def __init__(self):
        super().__init__(
            "The document processing deadline was exceeded.", "processing_timeout", 504
        )
