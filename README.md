This project presents a multi-agent, NVIDIA Agent Intelligence toolkit-powered diagnostic assistant designed to generate professional medical reports/prescriptions based on patient data for diabetes risk assessment. Built using NVIDIA’s Agentic AI infrastructure—including the NeMo framework and NIM APIs—the system demonstrates intelligent planning, multimodal input handling, and contextual reasoning through retrieval-augmented generation (RAG).

Key features:
1. Work on multi-modal data: allows users to input structured patient information or upload text/image files
2. The backend is driven by a primary planning agent (AgentWithMemoryAndTools) that leverages LangChain-based embeddings and FAISS to retrieve contextually relevant medical knowledge from a document index
3. Construct a detailed prompt that is submitted to NVIDIA’s Neva-22B NIM model to generate a comprehensive medical report
4. The assistant employs a modular agent architecture, where the main agent delegates file-saving responsibilities to a secondary FileWriterAgent
5.  Memory of all interactions is persistently stored and displayed, providing traceability of diagnostic insights

![Screenshot from 2025-05-23 09-20-10](https://github.com/user-attachments/assets/72c03f22-ebed-4224-889d-c20fc3a333f8)


# Virtual-Diagnostic-Assitant-for-Diabetes-Risk-Assesment
pip install streamlit sentence-transformers faiss-cpu PyPDF2 nim
pip install langchain-openai langchain-community langchain sentence-transformers faiss-cpu streamlit
pip install pytesseract pillow
pip install nemo-agent
pip install streamlit
pip install streamlit sentence-transformers faiss-cpu PyPDF2 nim
 pip install langchain-openai langchain-community langchain sentence-transformers faiss-cpu streamlit
 pip install langchain langchain-openai openai
 
