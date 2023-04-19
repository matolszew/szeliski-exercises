import streamlit as st
import numpy as np
import cv2


def local_histograms(img, patch_size):
    histograms = np.zeros((2 + img.shape[0] // patch_size, 2 + img.shape[1] // patch_size, 256))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ih = i // patch_size
            jh = j // patch_size
            s = (i % patch_size) / patch_size
            t = (j % patch_size) / patch_size
            histograms[ih, jh, img[i, j]] += (1 - s) * (1 - t)
            histograms[ih + 1, jh, img[i, j]] += s * (1 - t)
            histograms[ih, jh + 1, img[i, j]] += (1 - s) * t
            histograms[ih + 1, jh + 1, img[i, j]] += s * t

    return histograms


def cumulative_distributions(histograms):
    cdfs = np.zeros_like(histograms)
    for i in range(histograms.shape[0]):
        for j in range(histograms.shape[1]):
            cdfs[i, j] = np.cumsum(histograms[i, j]) / np.sum(histograms[i, j])
    return cdfs


def local_histogram_eq(img, cdfs, patch_size):
    equalized_img = np.zeros_like(img, dtype=np.float32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ih = i // patch_size
            jh = j // patch_size
            s = (i % patch_size) / patch_size
            t = (j % patch_size) / patch_size
            equalized_img[i, j] += cdfs[ih, jh, img[i, j]] * (1 - s) * (1 - t)
            equalized_img[i, j] += cdfs[ih + 1, jh, img[i, j]] * s * (1 - t)
            equalized_img[i, j] += cdfs[ih, jh + 1, img[i, j]] * (1 - s) * t
            equalized_img[i, j] += cdfs[ih + 1, jh + 1, img[i, j]] * s * t
    return np.round(255 * equalized_img).astype(np.uint8)


def local_histogram_eq_page():
    uploaded_file = st.file_uploader("Choose an image...")
    if uploaded_file is not None:
        file_bytes = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, caption="Original image", channels="BGR")
        image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        luminance = image_hsv[:, :, 2]
        patch_size = st.sidebar.slider(
            "Patch size",
            min_value=64,
            max_value=min(luminance.shape),
            value=min(luminance.shape) // 4,
            step=1
        )
        histograms = local_histograms(luminance, patch_size)
        cdfs = cumulative_distributions(histograms)
        equalized_luminance = local_histogram_eq(luminance, cdfs, patch_size)
        equlized_image_hsv = np.concatenate(
            (image_hsv[:, :, 0:2], equalized_luminance[:, :, None]),
            axis=2
        )
        equlized_image = cv2.cvtColor(equlized_image_hsv, cv2.COLOR_HSV2BGR)
        st.image(equlized_image, caption="Equalized image", channels="BGR")


if __name__ == "__main__":
    st.title("Local histogram equalization")
    local_histogram_eq_page()
