# Image Processing App with Edge Detection

A user-friendly Streamlit application that demonstrates advanced image processing techniques, specifically bilateral filtering and Canny edge detection. This tool allows you to upload images and experiment with different parameters to understand how these computer vision algorithms work in practice.

## What This Application Does

This application takes you through a complete image processing pipeline that transforms photographs into clean edge representations. The process involves three key stages that work together to produce high-quality edge detection results:

**Stage 1: Grayscale Conversion** - The uploaded image is first converted to grayscale, which simplifies processing by reducing the complexity from three color channels (RGB) to a single intensity channel. This step is essential because edge detection algorithms work on intensity gradients rather than color variations.

**Stage 2: Bilateral Filtering** - Before detecting edges, the image undergoes bilateral filtering, which is a sophisticated noise reduction technique. Unlike simple blur filters that smooth everything uniformly, bilateral filtering preserves important edges while reducing noise. It accomplishes this by considering both the spatial distance between pixels and the intensity difference between them, making it ideal for preparing images for edge detection.

**Stage 3: Canny Edge Detection** - The final stage applies the Canny edge detection algorithm, which identifies edges by finding areas where the image intensity changes rapidly. The algorithm uses two thresholds to determine which edges are significant enough to keep, resulting in clean, thin edge lines that outline the important features in your image.

## Installation and Setup

Before running the application, you'll need to set up your Python environment with the required dependencies. This process involves installing several powerful libraries that handle different aspects of image processing and web interface creation.

First, ensure you have Python 3.7 or higher installed on your system. Then, install the required packages using pip:

```bash
pip install -r requirements.txt
```

The application depends on four key libraries, each serving a specific purpose:

- **Streamlit** creates the interactive web interface that makes the application easy to use
- **OpenCV (opencv-python-headless)** provides the core image processing algorithms including bilateral filtering and Canny edge detection
- **Pillow (PIL)** handles image file reading and format conversions
- **NumPy** manages the underlying array operations that represent images as numerical data

## Running the Application

Once you have installed the dependencies, launching the application is straightforward:

```bash
streamlit run main.py
```

This command starts a local web server and automatically opens your default browser to display the application interface. If the browser doesn't open automatically, you can manually navigate to the URL shown in your terminal, typically `http://localhost:8501`.

## How to Use the Application

The application interface is designed to be intuitive while providing educational insight into how image processing parameters affect the final results.

**Step 1: Upload Your Image** - Use the file uploader to select a PNG, JPG, or JPEG image from your computer. The application will immediately process and display your image in grayscale format.

**Step 2: Adjust Bilateral Filter Parameters** - The left column contains three sliders that control the bilateral filtering process:

- **Diameter** controls the size of the neighborhood area used for filtering. Larger values include more surrounding pixels in the calculation, creating stronger smoothing effects but requiring more processing time.
- **Sigma Color** determines how much influence pixels with different intensities have on each other. Higher values mean that pixels with quite different intensities will influence each other more, resulting in more aggressive noise reduction.
- **Sigma Space** controls how much influence distant pixels have within the neighborhood. Larger values mean that farther pixels will influence the computation more, as long as their colors are close enough.

**Step 3: Fine-tune Canny Edge Detection** - The right column provides two critical thresholds for the Canny algorithm:

- **Low Threshold** sets the minimum gradient strength required for a pixel to be considered a potential edge. Pixels with gradients below this value are definitely not edges.
- **High Threshold** defines the minimum gradient strength for a pixel to be immediately classified as an edge. Pixels above this threshold are definitely edges.

The Canny algorithm uses a technique called hysteresis thresholding, where pixels between the low and high thresholds are only considered edges if they connect to pixels that are definitely edges.

**Step 4: View Results** - The application displays three images side by side, allowing you to see the effect of each processing stage. This visualization helps you understand how the bilateral filter prepares the image for edge detection and how the Canny algorithm responds to your parameter choices.

**Step 5: Export Your Results** - Once you're satisfied with the edge detection results, you can save your work in two formats:

- **PNG Format** preserves the edge image as a raster graphic, perfect for further image editing or analysis
- **SVG Format** converts the edges into scalable vector paths, ideal for graphic design applications or when you need resolution-independent results

## Understanding the Technical Details

The application implements a carefully designed processing pipeline that demonstrates important computer vision concepts.

**Bilateral Filtering Theory** - This algorithm solves a common problem in image processing: how to reduce noise without destroying important edges. Traditional smoothing filters blur everything equally, but bilateral filtering uses a weighted average that considers both spatial proximity and intensity similarity. The mathematical foundation involves two Gaussian distributions multiplied together, one for spatial distance and one for intensity difference.

**Canny Edge Detection Process** - The Canny algorithm is considered one of the most effective edge detectors because it optimizes three criteria: good detection (finding real edges), good localization (placing edges accurately), and single response (avoiding multiple responses to single edges). The algorithm applies Gaussian smoothing, calculates gradients, performs non-maximum suppression to thin edges, and finally uses double thresholding with hysteresis tracking.

**SVG Generation Approach** - The application converts edge pixels into vector paths by finding contours in the binary edge image and translating them into SVG path elements. This process demonstrates how raster images can be converted to scalable vector graphics, though the quality depends on the clarity and continuity of the detected edges.

## Parameter Tuning Guidelines

Learning to adjust the parameters effectively requires understanding how each setting influences the final result.

For the bilateral filter, start with moderate values and observe how changes affect noise reduction versus edge preservation. If your image is very noisy, increase the sigma values, but be careful not to over-smooth important details. The diameter parameter has the most dramatic effect on processing time, so use the smallest value that gives acceptable results.

For Canny edge detection, the relationship between the two thresholds is crucial. A good starting point is to set the high threshold at roughly twice the low threshold value. If you're getting too many weak edges, increase the low threshold. If important edges are missing, decrease the high threshold. Remember that the optimal values depend heavily on your specific image's contrast and content.

## File Structure

```
image-processing-app/
├── main.py              # Main application code with Streamlit interface
├── requirements.txt     # Python package dependencies
└── README.md           # This documentation file
```

## Technical Requirements

- Python 3.7 or higher
- At least 4GB of RAM for processing larger images
- Modern web browser for the Streamlit interface
- Sufficient storage space for saving processed images

This application serves as both a practical tool for edge detection and an educational platform for understanding fundamental computer vision algorithms. Experiment with different images and parameter combinations to develop an intuitive understanding of how these powerful techniques work in practice.
