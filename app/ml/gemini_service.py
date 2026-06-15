import os
import google.generativeai as genai
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv
from app.ml.rag_service import RAGService

# Load environment variables (like GEMINI_API_KEY)
load_dotenv()

class GeminiClinicalAssistant:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.is_configured = bool(self.api_key)
        
        if self.is_configured:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
        else:
            self.model = None

        self.rag_service = RAGService()
        
        self.system_prompt = (
            "You are an expert clinical AI assistant integrated into the DPCT (Dual-Path Clinical Transformer) sepsis detection system. "
            "Your role is to help clinicians interpret model predictions and answer clinical questions based strictly on provided guidelines. "
            "Always be concise, professional, and highlight critical clinical indicators. "
            "If discussing a prediction, refer to the patient context and the clinical threshold gates triggered. "
            "WARNING: Always include a disclaimer that you are an AI assistant and clinical judgment must take precedence."
        )

    def explain_prediction(self, query: str, patient_context: Optional[Dict[str, Any]] = None, history: Optional[List[Dict[str, str]]] = None) -> str:
        if not self.is_configured:
            return "Configuration Error: GEMINI_API_KEY is not set in the environment. Please add it to use the AI Clinical Assistant."

        # Retrieve relevant clinical context via RAG
        rag_context = self.rag_service.retrieve_context(query)

        # Build the full prompt
        prompt = f"{self.system_prompt}\n\n"
        
        if patient_context:
            prompt += "--- PATIENT PREDICTION CONTEXT ---\n"
            prompt += f"Model Probability: {patient_context.get('probability', 'Unknown')}\n"
            prompt += f"Risk Band: {patient_context.get('risk_band', 'Unknown')}\n"
            prompt += f"Attention Peak Hour: {patient_context.get('peak_hour', 'Unknown')}\n"
            
            flags = patient_context.get('clinical_flags', [])
            if flags:
                prompt += f"Triggered Clinical Gates: {', '.join(flags)}\n"
            
            vitals = patient_context.get('vitals', {})
            if vitals:
                prompt += "Current Vitals:\n"
                for k, v in vitals.items():
                    prompt += f"- {k}: {v}\n"
            prompt += "----------------------------------\n\n"

        prompt += "--- RELEVANT CLINICAL KNOWLEDGE ---\n"
        prompt += f"{rag_context}\n"
        prompt += "-----------------------------------\n\n"
        
        if history:
            prompt += "--- CONVERSATION HISTORY ---\n"
            for msg in history:
                role = "Clinician" if msg["role"] == "user" else "Assistant"
                prompt += f"{role}: {msg['content']}\n"
            prompt += "----------------------------\n\n"
        
        prompt += f"Clinician Query: {query}\n"
        prompt += "Response:"

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error communicating with Gemini API: {str(e)}"
