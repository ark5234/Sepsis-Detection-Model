from __future__ import annotations

import torch
import uvicorn


if __name__ == "__main__":
    print("===================================================================")
    print("Starting DPCT Inference Server...")
    print("NOTE: The very first startup will silently download the language")
    print("model for the Gemini Assistant (approx 90MB). This may take a few")
    print("minutes. Please do not close the window.")
    print("===================================================================")
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000)
