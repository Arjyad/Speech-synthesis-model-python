import streamlit as st
import requests

# Custom CSS for background color
st.markdown(
    """
    <style>
    .stApp {
        background-color: #87CEEB;  /* Sky Blue */
        padding: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Deepfake Generator")

selected_model = st.selectbox("Select Model", ["Model 1", "Model 2"])

source_file = st.file_uploader("Upload Source Voice File (e.g., the voice you want to transform)", type=["wav"])
target_file = st.file_uploader("Upload Target Voice File (e.g., the voice characteristics to apply)", type=["wav"])

# time.sleep(5) 

if st.button("Generate Deepfake"):
    if source_file and target_file:
        try:
            files = {
                'source_file': source_file,
                'target_file': target_file
            }

            response = requests.post("http://localhost:5000/convert", files=files)
            response.raise_for_status()
            
            result = response.json()
            output_file_url = result.get('outputFileUrl')

            if output_file_url:
                st.success("Deepfake generation successful!")
                st.markdown(f"[Download Output]({output_file_url})")
            else:
                st.error("Error in deepfake generation: No output file URL found.")

        except requests.exceptions.RequestException as e:
            st.error(f"Error uploading file: {e}")

    else:
        st.warning("Please upload both source and target files.")
