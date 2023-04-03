import streamlit as st
import numpy as np
import cv2


def cumulative_distribution(histogram):
    cdf = np.zeros_like(histogram)
    cdf[0] = histogram[0]
    for i, h in enumerate(histogram[1:]):
        cdf[i + 1] = cdf[i] + h

    cdf /= np.sum(histogram)

    return cdf


def equalize_histogram(image, cdf, punch, local_gain_limit):
    equalized_image = np.zeros_like(image, dtype=np.uint8)
    local_gain_reserve = 0
    last_val = 0
    for i, val in enumerate(cdf):
        val += local_gain_reserve
        dv = val - last_val
        if dv > local_gain_limit:
            local_gain_reserve = dv - local_gain_limit
            val = last_val + local_gain_limit
        else:
            local_gain_reserve = 0
        mask = image == i
        if val < punch:
            equalized_image[mask] = 0
        elif val > 1 - punch:
            equalized_image[mask] = 255
        else:
            equalized_image[mask] = np.round(255 * val)
        last_val = val
    return equalized_image


def histogram_equalization_page():
    uploaded_file = st.file_uploader("Choose a image")
    punch = st.sidebar.slider(
        '"punch" [%]',
        min_value=0,
        max_value=20,
        value=0) / 100
    local_gain_limit = st.sidebar.slider(
        'local gain limit',
        min_value=1e-6,
        max_value=0.1,
        value=0.1,
        step=1e-6
    )
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        col1, col2 = st.columns(2)

        col1.image(image, caption="Original image", channels="BGR")
        col1.image(image_gray, caption="Original grayscale image")
        histogram = cv2.calcHist([image_gray], [0], None, [256], [0, 256])[:, 0]
        col1.bar_chart(histogram)
        cdf = cumulative_distribution(histogram)
        col1.line_chart(cdf)

        equalized_grey_image = equalize_histogram(image_gray, cdf, punch, local_gain_limit)
        equalized_histogram = cv2.calcHist([equalized_grey_image], [0], None, [256], [0, 256])[:, 0]
        equalized_cdf = cumulative_distribution(equalized_histogram)

        eqalized_image = np.zeros_like(image, dtype=np.float32)
        luma_ratio = equalized_grey_image / image_gray
        for axis in range(3):
            color_ratio = image[:, :, axis] / np.sum(image, axis=2)
            eqalized_image[:, :, axis] = luma_ratio * color_ratio

        col2.image(eqalized_image, caption="Equalized image", channels="BGR")
        col2.image(equalized_grey_image, caption="Image after histogram equalization")
        col2.bar_chart(equalized_histogram)
        col2.line_chart(equalized_cdf)


if __name__ == "__main__":
    st.title("Histogram equalization")
    histogram_equalization_page()
