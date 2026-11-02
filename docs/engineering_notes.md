# Engineering notes: explain the system from first principles

## OCR, OMR, and native text

A digital PDF may already contain character codes and coordinates. A scan contains pixels. OCR estimates characters from those pixels; it does not recover a guaranteed original. OMR answers a smaller visual question: whether a particular region is marked. A checkbox can be detected correctly while its state is wrong, which is why those metrics are separate.

The native-text path avoids inference but still renders a page for checkbox inspection. A usable-looking PDF text layer can disagree with the visible page; the current heuristic cannot prove their semantic agreement. Hybrid OCR merges text sources conservatively, and complex/overlapping layouts remain a documented limitation.

Rendering cost and memory grow with pixel count. Doubling DPI approximately quadruples pixels. The pipeline handles one page at a time and checks the expected render dimensions before allocating the image. A small compressed input can expand into many pixels, so byte and pixel limits solve different problems.

## Preprocessing

Grayscale removes color channels; resizing changes sampling; denoising suppresses local variation; thresholding converts intensities to foreground/background; contrast enhancement redistributes intensity locally. Each can remove useful information as well as noise. Preserve the original and compare identical pages under declared profiles.

Rotation and deskew change geometry. Keep the forward transform, invert it for exported evidence boxes, and establish reading order in the corrected image. A correctly recognized line becomes useless if later sorting scrambles its order.

Most image operations are O(W×H) time and memory. Deskew's projection search repeats work over a bounded angle grid, so it is approximately O(A×W×H), with the search image resized to control cost. These are engineering cost estimates, not measured latency claims.

## Classical classification

TF-IDF represents a document as a sparse vector: terms frequent in one document but uncommon across training documents receive larger weights. LinearSVC learns separating scores for the three categories. Inference is a sparse dot product; its cost depends mainly on nonzero features and the number of classes.

Fit vocabulary and inverse-document frequencies only on training data. The validation margin threshold allows abstention. A large score is not a calibrated probability and is not evidence that an unfamiliar document belongs to a supported category.

## Rules and statistical NER

A rule explicitly describes a pattern, such as a labeled patient line or dosage phrase inside an instruction line. This is useful when formats are constrained. A statistical NER model learns token-context patterns from examples. The project's spaCy pipeline is trained from a blank English pipeline; it is not a pretrained medical expert.

NER produces spans, not complete medication objects. Deterministic assembly groups entities and attaches the full instruction line. This mixture is explicitly disclosed in comparisons. Labels for repeated strengths/doses must stay inside their own medication blocks; otherwise training examples contradict their intended structure.

The rule patterns are bounded line-oriented expressions rather than unconstrained nested regular expressions. Processing scales roughly with text length times a small number of patterns. NER cost grows mainly with token count and model configuration. Time the actual pipeline rather than treating these approximations as hardware benchmarks.

## Structured local LLM extraction

A JSON schema constrains shape and scalar types. It does not prove that the model read the correct patient, copied the right dose, or quoted the proper role. Nulls, exact quotes, page numbers, independent source checks, role checks, and held-out evaluation are separate controls.

Model weights, runtime buffers, and context/KV state all consume memory. Keep context bounded and process one local request at a time. Longer output also costs time: a prescription with three medications and repeated citations can exceed a budget that works for a one-medication example. Record cold load separately from warm generation.

## Metrics by hand

CER = character edit distance / reference characters. WER uses words instead of characters. For reference `500 mg` and hypothesis `500 mcg`, one inserted character gives CER = 1/6, while one word substitution gives WER = 1/2. A small average text error can still change a crucial field, so field exact match is necessary too.

Precision asks how many emitted entities are correct; recall asks how many reference entities were recovered. F1 balances the two. Field-path exact match additionally checks medication grouping. Missing-value false fills expose a model that invents plausible-looking values, even if many ordinary fields are correct.

The Levenshtein implementation takes O(n×m) time and O(min(n,m)) memory. Corpus CER/WER sum edits and reference lengths before division; averaging page percentages would over-weight short pages.

## Human review and honest claims

Review flags are named decisions, not probabilities. The weakest matching OCR span can flag an otherwise high-scoring page; ambiguous frequencies and explicitly uncertain medication names remain unchanged. Human edits live beside the original predictions, with notes and fresh validation.

The corpus shares a narrow vocabulary and consistent labels. Holding out a layout family is stronger than randomly splitting near-identical pages, but only one unseen family per category remains a small test. Describe the supported synthetic conditions and sample counts whenever reporting a score. No result here establishes real clinical accuracy, treatment safety, handwriting recognition, or compliance.
