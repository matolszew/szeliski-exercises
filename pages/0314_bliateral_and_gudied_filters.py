import streamlit as st
import numpy as np
import cv2


EXERCISE_DESC = """
Implement or download code for bilateral and/or guided image filtering and use this to implement \
some image enhancement or processing application, such as those described in Section 3.3.2
"""


def get_filter_params(filter_type):
    match filter_type:
        case "Bilateral":
            params = {
                "d": st.sidebar.slider("Diameter", 1, 10, 1),
                "sigmaColor": st.sidebar.slider("Color", 1, 200, 1),
                "sigmaSpace": st.sidebar.slider("Space", 1, 200, 1)
            }
        case "Guided":
            raise NotImplementedError("Guided filter not implemented")
    return params


def filters_page():
    uploaded_file = st.file_uploader("Choose an image...")
    filter_type = st.sidebar.radio("Select a filter type", ("Bilateral", "Guided"))
    params = get_filter_params(filter_type)

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, caption="Uploaded Image", channels="BGR")

        match filter_type:
            case "Bilateral":
                image = cv2.bilateralFilter(image, **params)
            case "Guided":
                st.write("Guided Filter")

        st.image(image, caption="Filtered Image", channels="BGR")



if __name__ == "__main__":
    st.title("Bilateral and guided image filters")
    st.caption(EXERCISE_DESC)
    filters_page()
