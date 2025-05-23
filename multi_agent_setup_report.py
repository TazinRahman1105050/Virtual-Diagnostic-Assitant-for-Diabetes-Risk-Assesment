# === The Backend file for diagnostic app project ===
import os
import requests
import faiss
import pickle
import numpy as np
from datetime import datetime
from nemo_agent import NemoAgent
from langchain_openai import OpenAIEmbeddings

### Get the context using RAG ###

embedding_model = OpenAIEmbeddings()

###Call NVIDIA NIM ###
def call_nim(query, retrieved_context, model="nvidia/neva-22b"):
    api_key = os.getenv("NIM_API_KEY")
    if not api_key:
        raise ValueError("NIM_API_KEY environment variable not set.")

    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful medical assistant."},
            {"role": "user", "content": f"{retrieved_context}\n\nQuestion: {query}"}
        ]
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code != 200:
        raise Exception(f"NIM request failed [{response.status_code}]: {response.text}")
    return response.json()["choices"][0]["message"]["content"]

### Retrieve chunks of context based on query ###
def retrieve_chunks(query, k=3):
    index_path = "RAG/data/index/docs.faiss"
    chunks_path = "RAG/data/index/docs.pkl"

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError("Missing FAISS index or chunks file.")

    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    query_embedding = embedding_model.embed_query(query)
    query_vector = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_vector, k)
    relevant_chunks = [chunks[i] for i in indices[0]]
    return "\n\n".join(relevant_chunks)
### Planning Agent with report writing agent (FileWriterAgent) ###
class AgentWithMemoryAndTools(NemoAgent):
    def __init__(self, task, api_key):
        super().__init__(task, api_key)
        self.memory = []
        self.writer_agent = FileWriterAgent("write_prescription", api_key)

    def run(self, patient_data):
        report = self.generate_report(patient_data)

        patient_name = patient_data.get("name", "Unknown").strip().replace(" ", "_").replace("/", "_")
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"MedicalReport_{patient_name}_{date_str}.txt"

        self.writer_agent.write_to_file(report, filename=filename)

        self.memory.append({"input": patient_data, "report": report, "filename": filename})

        return [report]

    def generate_report(self, data):
        if "raw_text" in data:
            return f"""# Generated Medical Report from Uploaded Text\n\n{data['raw_text']}\n\n---\n\n**Note**: This report was generated from unstructured input. Further analysis may be needed."""

        name = data.get("name", "Not specified")
        age = data.get("age", "Not specified")
        gender = data.get("gender", "Not specified")
        symptoms = data.get("symptoms", "")
        additional_info = data.get("additional_info", "")

        full_query = f"Symptoms: {symptoms}\nAdditional Info: {additional_info}"
        retrieved_context = retrieve_chunks(full_query)

        prompt = f"""
You are a medical assistant AI. Generate a complete, well-structured medical report.
Include:
- Patient Information
- Presenting Symptoms
- Clinical Findings
- Interpretation
- Diagnosis
- Recommendations
- Follow-up Plan

Patient Info:
- Name: {name}
- Age: {age}
- Gender: {gender}
- Symptoms: {symptoms}
- Additional Info: {additional_info}

Relevant Medical Context:
{retrieved_context}

Make the tone professional and use Markdown formatting.
"""

        try:
            report = call_nim(query=prompt, retrieved_context=retrieved_context)
        except Exception as e:
            report = f"⚠️ Error: {e}\nPrompt was:\n{prompt}"

        return report

    def get_memory(self):
        return self.memory

class FileWriterAgent(NemoAgent):
    def __init__(self, task, api_key):
        super().__init__(task, api_key)

    def write_to_file(self, content, filename="generated_prescription.txt"):
        with open(filename, "w") as f:
            f.write(content)

def get_agent():
    task = "generate_medical_report"
    api_key = os.getenv("NEMO_API_KEY")
    return AgentWithMemoryAndTools(task=task, api_key=api_key)
