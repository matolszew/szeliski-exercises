import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2


def get_rect_points(rect):
    p1 = np.array([[rect["left"], rect["top"]]])
    points = np.repeat(p1, 4, axis=0)
    points[1:3, 0] += rect["width"]
    points[2:4, 1] += rect["height"]

    return points


def homogeneous_points(points):
    return np.hstack((points, np.ones((points.shape[0], 1))))


def inhomogeneous_points(points):
    points[:, 0] /= points[:, 2]
    points[:, 1] /= points[:, 2]
    return points[:, :2]


def opencv_polylines_points(points):
    return np.int32(points.reshape((points.shape[0], 1, 2)))


def draw_rectangle(points, canvas_width, canvas_height):
    img = 255 * np.ones((canvas_height, canvas_width, 3), dtype=np.uint8)
    cv2.polylines(img, opencv_polylines_points(points), True, (255, 0, 0), 3, cv2.LINE_AA)
    return img


def translate_points(points, x_translation, y_translation):
    T = np.array([
        [1, 0, x_translation],
        [0, 1, y_translation]
    ])
    translated_points = T @ homogeneous_points(points).T

    return translated_points.T


def rotate_points(points, x_translation, y_translation, theta):
    rect_center = np.mean(points, axis=0)
    points = points - rect_center
    theta = np.radians(theta)
    Rt = np.array([
        [np.cos(theta), -np.sin(theta), x_translation],
        [np.sin(theta), np.cos(theta), y_translation]
    ])
    rotated_points = np.transpose(Rt @ homogeneous_points(points).T)
    rotated_points += rect_center

    return rotated_points


def scaled_rotation(points, x_translation, y_translation, theta, scale_factor):
    rect_center = np.mean(points, axis=0)
    points = points - rect_center
    theta = np.radians(theta)
    sRt = np.array([
        [np.cos(theta), -np.sin(theta), x_translation],
        [np.sin(theta), np.cos(theta), y_translation]
    ])
    sRt[:, :2] *= scale_factor
    new_points = np.transpose(sRt @ homogeneous_points(points).T)
    new_points += rect_center

    return new_points


def affine_transformation(points, affine):
    rect_center = np.mean(points, axis=0)
    points = points - rect_center
    new_points = np.transpose(affine @ homogeneous_points(points).T)
    new_points += rect_center

    return new_points


def perspective_transformation(points, perspective):
    rect_center = np.mean(points, axis=0)
    points = points - rect_center
    new_points = inhomogeneous_points(np.transpose(perspective @ homogeneous_points(points).T))
    new_points += rect_center

    return new_points


def editor():
    canvas_height = 400
    canvas_width = 600

    deformation_mode = st.sidebar.selectbox(
        "Deformation mode:",
        ("translation", "rigid", "similarity", "affine", "perspective")
    )

    canvas_result = st_canvas(
        drawing_mode="rect",
        stroke_width=1,
        height=canvas_height,
        width=canvas_width,
    )
    if canvas_result.json_data and canvas_result.json_data["objects"]:
        first_rect = canvas_result.json_data["objects"][0]
        points = get_rect_points(first_rect)

        match deformation_mode:
            case "translation":
                x_translation = st.sidebar.slider(
                    "X translation:",
                    value=0,
                    min_value=-canvas_width // 2,
                    max_value=canvas_width // 2,
                    step=1
                )
                y_translation = st.sidebar.slider(
                    "Y translation:",
                    value=0,
                    min_value=-canvas_height // 2,
                    max_value=canvas_height // 2,
                    step=1)
                points = translate_points(points, x_translation, y_translation)
            case "rigid":
                x_translation = st.sidebar.slider(
                    "X translation:",
                    value=0,
                    min_value=-canvas_width // 2,
                    max_value=canvas_width // 2,
                    step=1
                )
                y_translation = st.sidebar.slider(
                    "Y translation:",
                    value=0,
                    min_value=-canvas_height // 2,
                    max_value=canvas_height // 2,
                    step=1)
                theta = st.sidebar.slider(
                    "Rotation",
                    value=0,
                    min_value=-180,
                    max_value=180,
                    step=1
                )
                points = rotate_points(points, x_translation, y_translation, theta)
            case "similarity":
                x_translation = st.sidebar.slider(
                    "X translation:",
                    value=0,
                    min_value=-canvas_width // 2,
                    max_value=canvas_width // 2,
                    step=1
                )
                y_translation = st.sidebar.slider(
                    "Y translation:",
                    value=0,
                    min_value=-canvas_height // 2,
                    max_value=canvas_height // 2,
                    step=1)
                theta = st.sidebar.slider(
                    "Rotation",
                    value=0,
                    min_value=-180,
                    max_value=180,
                    step=1
                )
                scale_factor = st.sidebar.slider(
                    "Scale factor",
                    value=1.0,
                    min_value=0.1,
                    max_value=5.0,
                    step=0.1
                )
                points = scaled_rotation(points, x_translation, y_translation, theta, scale_factor)
            case "affine":
                affine = np.array([
                    [1, 0, 0],
                    [0, 1, 0]
                ])
                affine = st.sidebar.experimental_data_editor(affine)
                points = affine_transformation(points, affine)
            case "perspective":
                perspective = np.eye(3)
                perspective = st.sidebar.experimental_data_editor(perspective)
                points = perspective_transformation(points, perspective)
        transformed_img = draw_rectangle(points, canvas_width, canvas_height)
        st.image(transformed_img)


if __name__ == "__main__":
    st.title("2D transform editor")
    editor()
