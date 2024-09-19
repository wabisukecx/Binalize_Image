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

        # Display original image
        st.image(gray_image, caption='Original Image', use_column_width=True)

        # Sliders for parameters
        d = st.slider('d', 1, 20, 9)
        sigma_color = st.slider('sigmaColor', 1, 150, 75)
        sigma_space = st.slider('sigmaSpace', 1, 150, 75)
        thresh = st.slider('thresh', 0, 255, 127)
        maximum = st.slider('maximum', 0, 255, 255)

        # Process image
        bilateral_filtered = cv2.bilateralFilter(gray_image, d, sigma_color, sigma_space)
        _, binary = cv2.threshold(bilateral_filtered, thresh, maximum, cv2.THRESH_BINARY)

        # Display processed image
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