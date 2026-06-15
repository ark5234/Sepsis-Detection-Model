import sys
import traceback

print("1. Starting import test...", flush=True)
try:
    print("2. Importing fastapi...", flush=True)
    import fastapi
    print("3. Importing torch...", flush=True)
    import torch
    print("4. Importing faiss...", flush=True)
    import faiss
    print("5. Importing sentence_transformers...", flush=True)
    import sentence_transformers
    print("6. Importing app.ml.rag_service...", flush=True)
    from app.ml.rag_service import RAGService
    print("7. Initializing RAGService...", flush=True)
    r = RAGService()
    print("8. Importing app.main...", flush=True)
    import app.main
    print("9. Success!", flush=True)
except BaseException as e:
    print("CRASHED CAUGHT:", flush=True)
    traceback.print_exc()
