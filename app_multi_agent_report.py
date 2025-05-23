### Frontend app file ###
import streamlit as st
import pathlib
import time
import os
from datetime import datetime
from PIL import Image
import pytesseract
from agent_setup_final import get_agent

### UI header Virtual Diagnostic Assistant for Diabetes Risk Assessment ###
st.set_page_config(page_title="Virtual Diagnostic Assistant for Diabetes Risk Assessment", layout="centered")
st.title("Virtual Diagnostic Assistant for Diabetes Risk Assessment")

### Ensure API key is available ###
if not os.getenv("NEMO_API_KEY"):
    st.error("❌ NEMO_API_KEY environment variable not set.")
    st.stop()

agent = get_agent()

### Optional: upload text or image ###
st.header("📎 Upload Patient Data (Optional)")
uploaded_txt = st.file_uploader("Upload a Text File", type=["txt"])
uploaded_img = st.file_uploader("Or Upload an Image File (OCR will be applied)", type=["png", "jpg", "jpeg"])
auto_extracted_data = {}

if uploaded_txt:
    content = uploaded_txt.read().decode("utf-8")
    st.text_area("📄 Extracted Text", content, height=150)
    auto_extracted_data["raw_text"] = content

elif uploaded_img:
    image = Image.open(uploaded_img)
    text = pytesseract.image_to_string(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.text_area("📄 OCR Result", text, height=150)
    auto_extracted_data["raw_text"] = text

### Structured input fallback ###
st.header("🧾 Enter Patient Info (If not using file)")
patient_name = st.text_input("Patient Name", placeholder="e.g., John Doe")
age = st.number_input("Age", min_value=0, max_value=120, value=55, step=1)
gender = st.selectbox("Gender", options=["Not specified", "Female", "Male", "Other"])
symptoms = st.text_area("Symptoms", placeholder="e.g., High blood sugar, fatigue")
additional_info = st.text_area("Additional Info (Optional)", placeholder="e.g., HbA1c 7.6%")

### Generate report ###
if st.button("Generate Report"):
    if "raw_text" in auto_extracted_data:
        data = {
            "name": patient_name.strip() or "Unnamed",
            "raw_text": auto_extracted_data["raw_text"]
        }
        with st.spinner("🧠 Generating report from file..."):
            outputs = agent.run(data)
    else:
        if not patient_name.strip() or not symptoms.strip():
            st.warning("Please fill in required fields.")
        else:
            data = {
                "name": patient_name.strip(),
                "age": age,
                "gender": gender,
                "symptoms": symptoms.strip(),
                "additional_info": additional_info.strip()
            }
            with st.spinner("🧠 Generating report from form..."):
                outputs = agent.run(data)

    report_text = outputs[0]
    st.markdown("### 📄 Generated Medical Report")
    st.markdown(report_text)

    # Save & Download
    safe_name = data["name"].replace(" ", "_")
    filename = f"MedicalReport_{safe_name}_{datetime.now().strftime('%Y-%m-%d')}.txt"
    with open(filename, "w") as f:
        f.write(report_text)
    with open(filename, "rb") as f:
        st.download_button("📥 Download Report", f, file_name=filename, mime="text/plain")

### Show memory ###
if hasattr(agent, "get_memory") and agent.get_memory():
    st.sidebar.header("🧠 Agent Memory")
    for idx, memory_item in enumerate(reversed(agent.get_memory())):
        st.sidebar.markdown(f"### Report #{len(agent.get_memory()) - idx}")
        info = memory_item["input"]
        st.sidebar.markdown(f"- **Name**: {info.get('name', 'N/A')}")
        st.sidebar.markdown(f"- **Age**: {info.get('age', 'N/A')}")
        st.sidebar.markdown(f"- **Gender**: {info.get('gender', 'N/A')}")
        st.sidebar.markdown(f"- **File**: `{memory_item['filename']}`")
        with st.sidebar.expander("View Report"):
            st.sidebar.text(memory_item["report"][:700] + "\n..." if len(memory_item["report"]) > 700 else memory_item["report"])