import streamlit as st
import numpy as np
import cv2
from collections import defaultdict


EXERCISE_DESC = """
Ex 3.12: Implement some softening, sharpening, and non-linear diffusion (selective sharpening or \
noise removal) filters, such as Gaussian, median, and bilateral (Section 3.3.1), as discussed in \
Section 3.4.2.

Take blurry or noisy images (shooting in low light is a good way to get both) and try to improve \
their appearance and legibility.
"""


def get_filter_params(filter_types):
    params = defaultdict(dict)
    for filter_type in filter_types:
        match filter_type:
            case "Gaussian":
                st.sidebar.write("Gaussian filter params:")

                kernel_width = st.sidebar.slider(
                    "kernel size width",
                    min_value=1,
                    max_value=19,
                    value=1,
                    step=2
                )
                kernel_height = st.sidebar.slider(
                    "kernel size height",
                    min_value=1,
                    max_value=19,
                    value=1,
                    step=2
                )
                params["gaussian"]["ksize"] = (kernel_width, kernel_height)
                params["gaussian"]["sigmaX"] = st.sidebar.slider(
                    "std in X direction",
                    min_value=0.0,
                    max_value=10.0,
                    value=0.0
                )
                params["gaussian"]["sigmaY"] = st.sidebar.slider(
                    "std in Y direction",
                    min_value=0.0,
                    max_value=10.0,
                    value=0.0
                )
            case "median":
                st.sidebar.write("Median filter params:")
                params["median"]["ksize"] = st.sidebar.slider(
                    "aperture linear size",
                    min_value=3,
                    max_value=25,
                    value=3,
                    step=2
                )
            case "bilateral":
                st.sidebar.write("Bilateral filter params:")
                params["bilateral"]["d"] = st.sidebar.slider(
                    "filter size",
                    min_value=1,
                    max_value=9,
                    value=1,
                    step=1
                )
                params["bilateral"]["sigmaColor"] = st.sidebar.slider(
                    "Sigma in color space",
                    min_value=1.0,
                    max_value=200.0,
                    value=1.0
                )
                params["bilateral"]["sigmaSpace"] = st.sidebar.slider(
                    "Sigma in coordinate space",
                    min_value=1.0,
                    max_value=200.0,
                    value=1.0
                )
    return params


def sbn_removal_page():
    uploaded_file = st.file_uploader("Choose a image")
    filter_types = st.sidebar.multiselect(
        "Select filters",
        ["Gaussian", "median", "bilateral"]
    )
    params = get_filter_params(filter_types)

    if uploaded_file:
        file_bytes = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, caption="Original image", channels="BGR")

        for filter_type in filter_types:
            match filter_type:
                case "Gaussian":
                    image = cv2.GaussianBlur(image, **params["gaussian"])
                case "median":
                    image = cv2.medianBlur(image, **params["median"])
                case "bilateral":
                    image = cv2.bilateralFilter(image, **params["bilateral"])

        st.image(image, caption="Filtered image", channels="BGR")


if __name__ == "__main__":
    st.title("Sharpening, blur and noise removal")
    st.caption(EXERCISE_DESC)
    sbn_removal_page()
