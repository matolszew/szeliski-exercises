import streamlit as st
import numpy as np
import cv2


def color_balance():
    uploaded_file = st.file_uploader("Choose a image")
    red_balance = st.slider("Red", min_value=0.0, max_value=5.0, value=1.0)
    green_balance = st.slider("Green", min_value=0.0, max_value=5.0, value=1.0)
    blue_balance = st.slider("Blue", min_value=0.0, max_value=5.0, value=1.0)
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) / 255
        image[:, :, 2] *= red_balance
        image[:, :, 1] *= green_balance
        image[:, :, 0] *= blue_balance
        image[image > 1] = 1
        st.image(image, caption="Original image", channels="BGR")


if __name__ == "__main__":
    st.title("Color balance")
    color_balance()
