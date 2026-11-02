"""Standalone worker: no imports from the main project's dependency environment."""

import json
import os
import sys
from importlib.metadata import version
from pathlib import Path


def restore_polygon(polygon, angle, width, height):
    import cv2
    import numpy as np

    if angle <= 0:
        return np.asarray(polygon, dtype=float)
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    c, s = abs(matrix[0, 0]), abs(matrix[0, 1])
    out_w, out_h = int(height * s + width * c), int(height * c + width * s)
    matrix[0, 2] += (out_w - width) / 2
    matrix[1, 2] += (out_h - height) / 2
    points = np.column_stack([polygon, np.ones(len(polygon))])
    return (np.linalg.inv(np.vstack([matrix, [0, 0, 1]])) @ points.T).T[:, :2]


def main():
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    # Running a file in this directory would otherwise shadow the native
    # `paddle` package with our neighboring adapter module paddle.py.
    sys.path = [
        entry for entry in sys.path if Path(entry).resolve() != Path(__file__).resolve().parent
    ]
    from paddleocr import PaddleOCR
    from PIL import Image

    source, output, detection, recognition = sys.argv[1:]
    with Image.open(source) as image:
        width, height = image.size
    engine = PaddleOCR(
        device="cpu",
        text_detection_model_name=detection,
        text_recognition_model_name=recognition,
        use_doc_orientation_classify=True,
        use_doc_unwarping=False,
        use_textline_orientation=True,
    )
    words = []
    for prediction in engine.predict(source):
        result = prediction.json
        if isinstance(result, str):
            result = json.loads(result)
        result = result.get("res", result)
        angle = result.get("doc_preprocessor_res", {}).get("angle", 0)
        for text, score, polygon in zip(
            result["rec_texts"], result["rec_scores"], result["rec_polys"], strict=True
        ):
            restored = restore_polygon(polygon, angle, width, height)
            xs, ys = restored[:, 0], restored[:, 1]
            words.append(
                {
                    "text": text,
                    "score": float(score),
                    "line": len(words),
                    "bbox": [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))],
                }
            )
    Path(output).write_text(
        json.dumps(
            {
                "words": words,
                "versions": {
                    "paddleocr": version("paddleocr"),
                    "paddlepaddle": version("paddlepaddle"),
                },
            }
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
