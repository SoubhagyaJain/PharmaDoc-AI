"""Versioned evidence-bearing contracts. Scores are NOT probabilities."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictStr, field_validator

Category = Literal["prescription", "clinical_form", "regulatory", "unknown"]
Box = tuple[float, float, float, float]
Method = Literal["rules", "ner", "llm", "human", "omr"]


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Evidence(Model):
    page_number: int = Field(ge=1)
    text: str
    bbox: Box | None = None

    @field_validator("bbox")
    @classmethod
    def valid_box(cls, box: Box | None) -> Box | None:
        if box is not None:
            x0, y0, x1, y1 = box
            if not all(0 <= v <= 1 for v in box) or x0 > x1 or y0 > y1:
                raise ValueError("Bounding boxes must be ordered and normalized to [0, 1].")
        return box


class Word(Model):
    text: str
    bbox: Box
    line: int = 0
    score: float | None = None
    score_source: str | None = None
    _valid_box = field_validator("bbox")(Evidence.valid_box.__func__)


class FieldResult(Model):
    value: StrictStr | StrictBool | None = None
    method: Method = "rules"
    evidence: list[Evidence] = Field(default_factory=list)
    status: Literal["extracted", "missing", "ambiguous", "invalid"] = "missing"
    raw_score: float | None = None
    score_source: str | None = None


class MedicationResult(Model):
    name: FieldResult = Field(default_factory=FieldResult)
    strength: FieldResult = Field(default_factory=FieldResult)
    administered_dose: FieldResult = Field(default_factory=FieldResult)
    instructions: FieldResult = Field(default_factory=FieldResult)
    frequency: FieldResult = Field(default_factory=FieldResult)


class Checkbox(Model):
    bbox: Box
    label: str | None = None
    state: Literal["checked", "unchecked", "ambiguous"]
    value: bool | None
    ink_ratio: float = Field(ge=0, le=1)
    _valid_box = field_validator("bbox")(Evidence.valid_box.__func__)


class PageResult(Model):
    page_number: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    route: Literal["digital", "ocr", "hybrid", "blank", "failed"]
    text: str = ""
    words: list[Word] = Field(default_factory=list)
    checkboxes: list[Checkbox] = Field(default_factory=list)
    image_coverage: float = Field(default=0, ge=0, le=1)
    issues: list[str] = Field(default_factory=list)


class ValidationIssue(Model):
    field: str
    code: str
    message: str


class ReviewCorrection(Model):
    value: StrictStr | StrictBool | None
    reason: str = Field(min_length=1)


class DocumentResult(Model):
    schema_version: str = "1.0"
    document_id: str
    document_category: Category = "unknown"
    classification_method: str = "keywords"
    classification_margin: float | None = None
    pages: list[PageResult] = Field(default_factory=list)
    fields: dict[str, FieldResult] = Field(default_factory=dict)
    medications: list[MedicationResult] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    validation_errors: list[ValidationIssue] = Field(default_factory=list)
    review_reasons: list[str] = Field(default_factory=list)
    human_review_required: bool = True
    status: Literal["complete", "partial", "failed"] = "complete"
    processing_duration_seconds: float = Field(default=0, ge=0)
    stage_timings: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    reviewed_fields: dict[str, ReviewCorrection] = Field(default_factory=dict)
