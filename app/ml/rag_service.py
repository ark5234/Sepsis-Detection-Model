from __future__ import annotations

import os
from typing import List, Dict, Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RAGService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
        # Clinical Knowledge Base
        self.documents = [
            # Sepsis-3 Guidelines
            "Sepsis is defined as life-threatening organ dysfunction caused by a dysregulated host response to infection (Sepsis-3).",
            "Organ dysfunction can be identified as an acute change in total SOFA score >= 2 points consequent to the infection.",
            "The baseline SOFA score should be assumed to be zero unless the patient is known to have preexisting (acute or chronic) organ dysfunction.",
            "Patients with suspected infection who are likely to have a prolonged ICU stay or to die in the hospital can be promptly identified at the bedside with qSOFA.",
            "The qSOFA score consists of three criteria: alteration in mental status, systolic blood pressure <= 100 mm Hg, or respiratory rate >= 22 /min.",
            "Septic shock is a subset of sepsis in which underlying circulatory and cellular/metabolic abnormalities are profound enough to substantially increase mortality.",
            "Patients with septic shock can be clinically identified by a vasopressor requirement to maintain a mean arterial pressure of 65 mm Hg or greater and serum lactate level greater than 2 mmol/L (>18 mg/dL) in the absence of hypovolemia.",
            
            # DPCT Model Context
            "The Dual-Path Clinical Transformer (DPCT) is a machine learning model for early sepsis detection.",
            "The DPCT uses a patient-level split to avoid temporal data leakage, evaluating on hourly ICU time series.",
            "The DPCT architecture splits data into a dense Vital Signs path and a sparse Laboratory Values path.",
            "Vital signs processed include HR (Heart Rate), O2Sat, Temp, SBP (Systolic BP), MAP (Mean Arterial Pressure), DBP (Diastolic BP), and Resp (Respiratory Rate).",
            "Laboratory values include sparse measurements like Lactate, WBC, Creatinine, Glucose, and others.",
            "The Time-Decay Lab Embedding penalizes stale lab measurements explicitly using an exponential time-decay mechanism based on hours elapsed since the last measurement.",
            "Bidirectional Cross-Attention allows the vital signs stream and laboratory stream to query one another before self-attention fusion.",
            "The Clinical Threshold Gate maps qSOFA/SOFA criteria into differentiable attention biases. It assigns high weights to MAP < 65 and Resp > 22.",
            "Clinical Attention Pooling provides an 'attention timeline' showing which ICU hour drove the sepsis prediction.",
            "The DPCT achieves an ROC-AUC of 0.9517 in 5-fold cross validation on the PhysioNet 2019 challenge dataset.",
            "A high optimal decision threshold (τ = 0.88) reduces false-positive alarm fatigue in clinical practice by ensuring high discriminative confidence.",
            "SHAP analysis reveals baseline models often rely on administrative proxies like ICU Length of Stay (ICULOS), whereas DPCT avoids these shortcuts."
        ]
        
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Build index
        embeddings = self.model.encode(self.documents)
        self.index.add(np.array(embeddings).astype("float32"))

    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """Retrieves top-k relevant documents for the given query."""
        query_embedding = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_embedding).astype("float32"), top_k)
        
        results = []
        for i in indices[0]:
            if 0 <= i < len(self.documents):
                results.append(self.documents[i])
                
        return "\n".join(f"- {res}" for res in results)
