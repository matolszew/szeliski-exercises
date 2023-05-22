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


def gaussian_kernels(kernel_size, sigma=1):
    l = np.linspace(-2 * np.ones(kernel_size), 2 * np.ones(kernel_size), kernel_size)
    x, y = l.T, -l
    exp = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    deg0 = -2 * x * exp
    deg90 = -2 * y * exp

    return deg0, deg90


def steerable_filters_page():
    uploaded_file = st.file_uploader("Choose a image")
    kernel_size = st.sidebar.slider("Kernel size", min_value=3, max_value=32, value=5, step=2)
    sigma = st.sidebar.slider("Sigma", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
    theta = st.sidebar.slider("Theta", min_value=0.0, max_value=2 * np.pi, value=0.0, step=0.01)
    g0, g90 = gaussian_kernels(kernel_size, sigma)

    if uploaded_file:
        file_bytes = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        st.image(image, caption='Uploaded Image.', channels="BGR")

        col1, col2, col3 = st.columns(3)
        with col1:
            g0_image = (g0 - g0.min()) / (g0.max() - g0.min())
            st.image(g0_image, caption='G0')
        with col2:
            g90_image = (g90 - g90.min()) / (g90.max() - g90.min())
            st.image(g90_image, caption='G90')
        with col3:
            theta_kernel = np.cos(theta) * g0 + np.sin(theta) * g90
            theta_image = (theta_kernel - theta_kernel.min()) / (theta_kernel.max() - theta_kernel.min())
            st.image(theta_image, caption='G_Theta')

        conv_img_0 = cv2.filter2D(image, -1, g0)
        conv_img_90 = cv2.filter2D(image, -1, g90)
        img_theta = np.cos(theta) * conv_img_0 + np.sin(theta) * conv_img_90
        img_theta = np.mean(img_theta, axis=2)
        img_theta_scled = (img_theta - img_theta.min()) / (img_theta.max() - img_theta.min())
        st.image(img_theta_scled, caption='G_Theta')


if __name__ == "__main__":
    st.title("Steerable filters")
    st.caption(EXERCISE_DESC)
    steerable_filters_page()
