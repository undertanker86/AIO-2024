import cv2
import numpy as np
from PIL import Image
import streamlit as st
import os

MODEL = "D:/AIO-2024-WORK/AIO-2024/module1-week4-streamlit_project/model/MobileNetSSD_deploy.caffemodel"
PROTOTXT = "D:/AIO-2024-WORK/AIO-2024/module1-week4-streamlit_project/model/MobileNetSSD_deploy.prototxt.txt"

if not os.path.isfile(MODEL) or not os.path.isfile(PROTOTXT):
    st.error("Model files are not found. Please check the paths.")
else:
    st.success("Model files are found and accessible.")


def process_image(image):
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
    )
    net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    net.setInput(blob)
    detections = net.forward()
    return detections


def annotate_image(
    image, detections, confidence_threshold=0.5
):
    # loop over the detections
    (h, w) = image.shape[:2]
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > confidence_threshold:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            # idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (s_x, s_y, e_x, e_y) = box.astype("int")
            cv2.rectangle(image, (s_x, s_y), (e_x, e_y), 70, 2)
    return image


def main():
    st.title('Object Detection for Images')
    file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])
    if file is not None:
        st.image(file, caption="Uploaded Image")

        image = Image.open(file)
        image = np.array(image)
        detections = process_image(image)
        processed_image = annotate_image(image, detections)
        st.image(processed_image, caption="Processed Image")


if __name__ == "__main__":
    main()
