import streamlit as st
import numpy as np
import cv2


EXERCISE_DESC = """
Ex 3.13: Steerable filters. Implement Freeman and Adelson's (1991) steerable filter algorithm. The \
input should be a grayscale or color image and the output should be a multi-banded image consisting\
of G0 and G90. The coefficients for the filters can be found in the paper by Freeman and Adelson \
(1991).

Test the various order filters on a number of images of your choice and see if you can reliably \
find corner and intersection features. These filters will be quite useful later to detect \
elongated structures, such as lines (Section 7.4).
"""


def steerable_filters_page():
    uploaded_file = st.file_uploader("Choose a image")

    if uploaded_file:
        file_bytes = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, caption='Uploaded Image.', use_column_width=True)

        # TODO: Implement steerable filters here.


if __name__ == "__main__":
    st.title("Steerable filters")
    st.caption(EXERCISE_DESC)
