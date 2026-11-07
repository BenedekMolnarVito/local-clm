"""Minimal int8 ONNX server for ownyourjev models (`ownyourjev serve` runs PyTorch only, too slow on this CPU).

Run: .venv-oyj/Scripts/python.exe oyj/serve_onnx.py models/oyj --host 127.0.0.1 --port 8701
One model per registry subdirectory; qid = directory name. Same /v1/systemone shape as TypeSafe.
"""
import argparse
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from ownyourjev.export import OnnxPredictor
from ownyourjev.spec import DecisionRequest


def create_app(registry, threads=4):
    preds = {d.name: OnnxPredictor(d, d / "model.int8.onnx", threads=threads)
             for d in sorted(Path(registry).iterdir()) if (d / "model.int8.onnx").is_file()}
    app = FastAPI()

    @app.get("/v1/models")
    def models():
        return {"data": [{"id": k} for k in preds]}

    @app.post("/v1/systemone")
    def systemone(req: DecisionRequest):
        t0 = time.perf_counter()
        answers = {}
        for qid, q in req.questions.items():
            p = preds.get(qid)
            if p is None or p.question.spec_hash() != q.spec_hash():  # never answer a question the model was not trained on
                raise HTTPException(404, detail=f"no trained model for question {qid!r}")
            answers[qid] = p.answer(req.state).model_dump()
        return {"answers": answers, "model": "ownyourjev-onnx",
                "usage": {"input_tokens": 0, "output_tokens": 0, "cost": 0.0, "latency_ms": (time.perf_counter() - t0) * 1000}}

    return app


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("registry")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8701)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    uvicorn.run(create_app(a.registry, a.threads), host=a.host, port=a.port, log_level="warning")
