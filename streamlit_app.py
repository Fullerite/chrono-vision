import base64
import os
from io import BytesIO

import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://backend:8000/process")

st.set_page_config(page_title="ChronoVision", layout="wide")

st.title("ChronoVision: Photo Colorization & Restoration")
st.markdown("Upload a grayscale image to colorize and restore it using our AI pipeline.")

uploaded_file = st.file_uploader("Choose a grayscale image", type=["jpg", "jpeg", "png"])


def decode_base64_image(base64_string):
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))


if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width=300)

    if st.button("Process Image"):
        with st.spinner("Processing... The GPU is warming up!"):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(API_URL, files=files)

                if response.status_code == 200:
                    data = response.json()

                    original = decode_base64_image(data['original'])
                    colorized = decode_base64_image(data['colorized'])
                    restored = decode_base64_image(data['restored'])

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.header("Original")
                        st.image(original, use_column_width=True)

                    with col2:
                        st.header("Colorized")
                        st.image(colorized, use_column_width=True)

                    with col3:
                        st.header("Restored")
                        st.image(restored, use_column_width=True)
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend. Ensure Docker is running.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
