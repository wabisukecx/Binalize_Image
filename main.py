import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import base64

def edge_to_svg(edges, width, height):
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    svg_string = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
    svg_string += '<path d="'
    
    for contour in contours:
        for i, point in enumerate(contour):
            x, y = point[0]
            if i == 0:
                svg_string += f'M{x},{y} '
            else:
                svg_string += f'L{x},{y} '
        svg_string += 'Z '
    
    svg_string += '" fill="none" stroke="black" stroke-width="1"/>'
    svg_string += '</svg>'
    
    return svg_string

def main():
    st.title('Image Processing App')

    # File uploader
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Convert to grayscale
        if len(image_array.shape) == 3:
            gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image_array

        # Sliders for parameters
        col1, col2 = st.columns(2)
        with col1:
            d = st.slider('Bilateral filter diameter', 5, 15, 5)
            sigma_color = st.slider('Bilateral filter sigmaColor', 50, 150, 75)
            sigma_space = st.slider('Bilateral filter sigmaSpace', 10, 150, 75)
        with col2:
            low_threshold = st.slider('Canny Low Threshold', 50, 150, 100)
            high_threshold = st.slider('Canny High Threshold', 150, 250, 200)

        # Process image
        # 1. Apply bilateral filter
        bilateral_filtered = cv2.bilateralFilter(gray_image, d, sigma_color, sigma_space)
        
        # 2. Apply Canny edge detection
        edges = cv2.Canny(bilateral_filtered, low_threshold, high_threshold)

        # Display original and processed images
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(gray_image, caption='Grayscale Image', use_column_width=True)
        with col2:
            st.image(bilateral_filtered, caption='Bilateral Filtered', use_column_width=True)
        with col3:
            st.image(edges, caption='Canny Edges', use_column_width=True)

        # Save buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button('Save as PNG'):
                edge_image = Image.fromarray(edges)
                buf = io.BytesIO()
                edge_image.save(buf, format='PNG')
                byte_im = buf.getvalue()

                st.download_button(
                    label="Download PNG",
                    data=byte_im,
                    file_name="canny_edges.png",
                    mime="image/png"
                )
        
        with col2:
            if st.button('Save as SVG'):
                svg_string = edge_to_svg(edges, edges.shape[1], edges.shape[0])
                b64 = base64.b64encode(svg_string.encode('utf-8')).decode('utf-8')
                href = f'data:image/svg+xml;base64,{b64}'
                st.markdown(f'<a href="{href}" download="canny_edges.svg">Download SVG</a>', unsafe_allow_html=True)

if __name__ == '__main__':
    main()
