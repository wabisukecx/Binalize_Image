import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def main():
    st.title('Image Processing App')

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Convert to grayscale if it's not already
        if len(image_array.shape) == 3:
            gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image_array

        # Sliders for parameters
        col1, col2 = st.columns(2)
        with col1:
            d = st.slider('d', 1, 20, 9)
            sigma_color = st.slider('sigmaColor', 1, 150, 75)
            sigma_space = st.slider('sigmaSpace', 1, 150, 75)
        with col2:
            thres = st.slider('thresh', 0, 255, 127)
            maximum = st.slider('maximum', 0, 255, 255)

        # Process image
        bilateral_filtered = cv2.bilateralFilter(gray_image, d, sigma_color, sigma_space)
        _, binary = cv2.threshold(bilateral_filtered, thres, maximum, cv2.THRESH_BINARY)

        # Display original and processed images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.image(gray_image, caption='Original Image', use_column_width=True)
        with col2:
            st.image(binary, caption='Processed Image', use_column_width=True)

        # Save button
        if st.button('Save Processed Image'):
            processed_image = Image.fromarray(binary)
            buf = io.BytesIO()
            processed_image.save(buf, format='PNG')
            byte_im = buf.getvalue()

            st.download_button(
                label="Download Processed Image",
                data=byte_im,
                file_name="processed_image.png",
                mime="image/png"
            )

if __name__ == '__main__':
    main()
