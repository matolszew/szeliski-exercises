import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial.distance import cdist


EXERCISE_DESC = """
Generate some random samples from a smoothly varying function and then implement and evaluate one \
or more data interpolation techniques.

1. Generate a “random” 1-D or 2-D function by adding together a small number of sinusoids or \
    Gaussians of random amplitudes and frequencies or scales.
2. Sample this function at a few dozen random locations.
3. Fit a function to these data points using one or more of the scattered data interpolation \
    techniques described in Section 4.1.
4. Measure the fitting error between the estimated and original functions at some set of location, \
    e.g., on a regular grid or at different random points.
5. Manually adjust any parameters your fitting algorithm may have to minimize the output sample \
    fitting error, or use an automated technique such as cross-validation.
6. Repeat this exercise with a new set of random input sample and output sample locations. Does the\
     optimal parameter change, and if so, by how much?
7. (Optional) Generate a piecewise-smooth test function by using different random parameters in \
    different parts of of your image. How much more difficult does the data fitting problem become?\
    Can you think of ways you might mitigate this?

Try to implement your algorithm in NumPy (or Matlab) using only array operations, in order to \
become more familiar with data-parallel programming and the linear algebra operators built into \
these systems. Use data visualization techniques such as those in Figures 4.3–4.6 to debug your \
algorithms and illustrate your results.
"""


def generate_sin(tab, tab_name):
    x = np.linspace(0, 512, 512)
    y = np.linspace(0, 512, 512)
    xx, yy = np.meshgrid(x, y)
    with tab:
        col1, col2, col3 = st.columns(3)
        with col1:
            A = st.slider(
                "amplitude",
                min_value=-5.0,
                max_value=5.0,
                value=1.0,
                key=f"{tab_name}_A"
            )
            fx = st.slider(
                "fx",
                min_value=-5,
                max_value=5,
                value=1,
                key=f"{tab_name}_fx"
            )
            fy = st.slider(
                "fy",
                min_value=-5,
                max_value=5,
                value=1,
                key=f"{tab_name}_fy"
            )
        vals = A * np.sin(2 * np.pi * (fx * xx + fy * yy))

        with col2:
            fig = plt.figure()
            plt.imshow(vals)
            st.pyplot(fig)

        with col3:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.plot_surface(xx, yy, vals, cmap=cm.coolwarm)
            st.pyplot(fig)

    return vals


def generate_gaussian(tab, tab_name):
    x = np.linspace(0, 511, 512)
    y = np.linspace(0, 511, 512)
    xx, yy = np.meshgrid(x, y)
    with tab:
        col1, col2, col3 = st.columns(3)
        with col1:
            A = st.slider(
                "amplitude",
                min_value=0.0,
                max_value=10.0,
                value=1.0,
                key=f"{tab_name}_A"
            )
            x0 = st.slider(
                "x0",
                min_value=0,
                max_value=512,
                value=256,
                key=f"{tab_name}_x0"
            )
            y0 = st.slider(
                "y0",
                min_value=0,
                max_value=512,
                value=256,
                key=f"{tab_name}_y0"
            )
            sigma_x = st.slider(
                "sigma_x",
                min_value=1,
                max_value=2000,
                value=500,
                key=f"{tab_name}_sigma_x"
            )
            sigma_y = st.slider(
                "sigma_y",
                min_value=1,
                max_value=2000,
                value=500,
                key=f"{tab_name}_sigma_y"
            )

        vals = A * np.exp(- ((xx - x0)**2) / sigma_x**2 - ((yy - y0)**2) / sigma_y**2)

        with col2:
            fig = plt.figure()
            plt.imshow(vals)
            st.pyplot(fig)

        with col3:
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            ax.plot_surface(xx, yy, vals, cmap=cm.coolwarm)
            st.pyplot(fig)

    return vals


def generate_functions(n_sin, n_gauss):
    x = np.linspace(0, 511, 512)
    y = np.linspace(0, 511, 512)
    xx, yy = np.meshgrid(x, y)
    gt = np.zeros_like(xx)
    with st.expander("Adjust function"):
        tab_names = [f"sin_{i}" for i in range(n_sin)]
        tab_names.extend([f"gaussian_{i}" for i in range(n_gauss)])
        tabs = st.tabs(tab_names)

        for tab, tab_name in zip(tabs, tab_names):
            if "sin" in tab_name:
                gt += generate_sin(tab, tab_name)
            elif "gaussian" in tab_name:
                gt += generate_gaussian(tab, tab_name)

    st.subheader("Generated function")
    col1, col2 = st.columns(2)
    with col1:
        fig = plt.figure()
        plt.imshow(gt, cmap=cm.coolwarm)
        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(xx, yy, gt, cmap=cm.coolwarm)
        st.pyplot(fig)

    return gt


def get_samples(gt, n_samples):
    sample_idx = np.random.randint(512, size=(2, n_samples))
    sample_values = gt[sample_idx[0], sample_idx[1]]

    st.subheader("Sampled function")
    col1, col2 = st.columns(2)
    with col1:
        fig = plt.figure()
        plt.scatter(sample_idx[0], sample_idx[1], c=sample_values, cmap=cm.coolwarm)
        plt.gca().invert_yaxis()
        st.pyplot(fig)

    with col2:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(sample_idx[0], sample_idx[1], sample_values, c=sample_values, cmap=cm.coolwarm)
        st.pyplot(fig)

    return sample_idx, sample_values


def get_basis_function_distance(radial_dist, basis_function, c):
    match basis_function:
        case "Gaussian":
            return np.exp(-(radial_dist**2) / c**2)
        case "Hardy multiquadratic":
            return np.sqrt(radial_dist**2 + c**2)
        case "Inverse multiquadratic":
            return 1 / np.sqrt(radial_dist**2 + c**2)
        case "Thin plate spline":
            return (radial_dist**2) * np.log(radial_dist)


def data_fitting_page():
    n_sin = st.sidebar.number_input(
        "No of sinus functions",
        min_value=0,
        value=1)
    n_gauss = st.sidebar.number_input(
        "No of gaussian functions",
        min_value=0,
        value=1)
    gt = generate_functions(n_sin, n_gauss)

    n_samples = st.sidebar.number_input(
        "No of samples",
        min_value=16,
        max_value=262144
    )
    inputs, targets = get_samples(gt, n_samples)

    basis_function = st.sidebar.selectbox(
        "basis function",
        ["Gaussian", "Hardy multiquadratic", "Inverse multiquadratic", "Thin plate spline"]
    )
    c = st.sidebar.number_input("c", min_value=0.1, value=100.0)
    radial_dist = cdist(inputs.T, inputs.T, metric="euclidean")
    basis_function_dist = get_basis_function_distance(radial_dist, basis_function, c)

    weights = np.linalg.inv(basis_function_dist) @ targets

    interpolated_data = np.empty((512, 512))
    for i in range(512):
        for j in range(512):
            radial_dist = cdist(inputs.T, [[i, j]])
            basis_function_dist = get_basis_function_distance(radial_dist, basis_function,c)
            interpolated_data[i, j] = weights @ basis_function_dist

    st.subheader("Interpolated function")
    col1, col2 = st.columns(2)
    with col1:
        fig = plt.figure()
        plt.imshow(interpolated_data, cmap=cm.coolwarm)
        st.pyplot(fig)

    with col2:
        x = np.linspace(0, 511, 512)
        y = np.linspace(0, 511, 512)
        xx, yy = np.meshgrid(x, y)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(xx, yy, interpolated_data, cmap=cm.coolwarm)
        st.pyplot(fig)


if __name__ == "__main__":
    st.title("Data fitting")
    with st.expander("Exercise description"):
        st.caption(EXERCISE_DESC)
    data_fitting_page()
