import sys
from pathlib import Path

from flask import Flask, Response, request

MODEL_DIR = Path("/opt/ml/model")
CODE_DIR = MODEL_DIR / "code"

sys.path.insert(0, str(CODE_DIR))
import inference  # noqa: E402

app = Flask(__name__)
artifact = None


def get_artifact():
    global artifact
    if artifact is None:
        artifact = inference.model_fn(str(MODEL_DIR))
    return artifact


@app.get("/ping")
def ping():
    try:
        get_artifact()
        return Response(status=200)
    except Exception as exc:
        return Response(str(exc), status=500, mimetype="text/plain")


@app.post("/invocations")
def invocations():
    try:
        content_type = request.content_type.split(";")[0] if request.content_type else "application/json"
        input_data = inference.input_fn(request.get_data(), content_type)
        prediction = inference.predict_fn(input_data, get_artifact())

        accept = request.headers.get("Accept", "application/json")
        body, output_type = inference.output_fn(prediction, accept)
        return Response(body, status=200, mimetype=output_type)
    except Exception as exc:
        return Response(str(exc), status=400, mimetype="text/plain")