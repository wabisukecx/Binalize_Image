# CLAUDE.md - AI Assistant Guide for Binalize_Image

## Project Overview

**Project Name:** Image Processing App with Edge Detection
**Type:** Streamlit web application
**Primary Language:** Python
**Main Purpose:** Interactive image processing tool demonstrating bilateral filtering and Canny edge detection

This is a lightweight, single-file Streamlit application that provides an educational and practical interface for experimenting with computer vision algorithms. The application processes uploaded images through a three-stage pipeline: grayscale conversion, bilateral filtering, and Canny edge detection.

## Repository Structure

```
Binalize_Image/
├── main.py              # Single-file Streamlit application (95 lines)
├── requirements.txt     # Python dependencies (4 packages)
├── README.md           # Comprehensive user documentation
└── CLAUDE.md           # This file - AI assistant guide
```

### File Descriptions

**main.py** (Lines 1-95)
- Entry point and entire application logic
- Key functions:
  - `edge_to_svg(edges, width, height)` (lines 8-26): Converts binary edge images to SVG format using OpenCV contour detection
  - `main()` (lines 28-95): Streamlit UI with image upload, parameter controls, processing pipeline, and export functionality
- No classes, minimal complexity, straightforward imperative style

**requirements.txt**
- `streamlit` - Web interface framework
- `opencv-python-headless` - Computer vision algorithms (bilateral filter, Canny edge detection, contour finding)
- `pillow` - Image file I/O
- `numpy` - Array operations for image data

## Code Architecture & Design Patterns

### Architecture Style
- **Monolithic single-file application** - All functionality in main.py
- **Procedural programming** - No classes or OOP patterns
- **Streamlit reactive model** - UI reruns on parameter changes
- **Pipeline processing** - Sequential transformation stages

### Key Technical Concepts

**Image Processing Pipeline:**
1. Input: Upload via Streamlit file uploader (PNG/JPG/JPEG)
2. Grayscale conversion: RGB → Single intensity channel (main.py:40-43)
3. Bilateral filtering: Noise reduction while preserving edges (main.py:57)
4. Canny edge detection: Gradient-based edge identification (main.py:60)
5. Output: Visual display + optional PNG/SVG export

**Parameter Controls:**
- Bilateral filter: diameter (5-15), sigmaColor (50-150), sigmaSpace (10-150)
- Canny detector: low_threshold (50-150), high_threshold (150-250)

**SVG Generation:**
- Uses `cv2.findContours()` to extract edge contours (main.py:9)
- Converts contours to SVG path commands (M for move, L for line, Z for close)
- Returns inline SVG string for download

## Development Workflows

### Running the Application

```bash
# Install dependencies
pip install -r requirements.txt

# Launch application
streamlit run main.py
```

The app will be accessible at `http://localhost:8501`

### Git Workflow

**Branch Naming Convention:**
- Feature branches follow pattern: `claude/claude-md-*`
- Example: `claude/claude-md-miv23b7b94ajjzx5-011tQmtiRgsA62rTVTJ4s6S2`

**Git Best Practices:**
- Always develop on designated Claude branches
- Use descriptive commit messages
- Push with: `git push -u origin <branch-name>`
- Retry on network failures with exponential backoff (2s, 4s, 8s, 16s)

### Testing
- No automated tests currently exist in the repository
- Manual testing via Streamlit interface is the current approach
- Consider adding tests for `edge_to_svg()` function if expanding functionality

## Key Conventions for AI Assistants

### When Modifying Code

1. **Preserve Simplicity**
   - This is intentionally a simple, educational application
   - Avoid over-engineering or adding unnecessary abstractions
   - Keep the single-file structure unless significant expansion is needed

2. **Maintain Processing Pipeline**
   - Changes should respect the three-stage pipeline (grayscale → bilateral → Canny)
   - Parameter ranges are carefully chosen for educational purposes
   - Don't break the real-time preview functionality

3. **Image Format Compatibility**
   - Support PNG, JPG, JPEG formats (main.py:32)
   - Handle both RGB and grayscale input images (main.py:40-43)
   - Ensure output formats (PNG/SVG) remain functional

4. **Streamlit Best Practices**
   - Use columns for organized layouts (main.py:46, 63, 72)
   - Maintain responsive design with `use_column_width=True`
   - Preserve download button patterns for exports

### Code Style Guidelines

- **Naming:** Use descriptive variable names (bilateral_filtered, edges, etc.)
- **Comments:** Minimal inline comments; rely on clear variable names
- **Formatting:** Standard Python formatting, 4-space indentation
- **Dependencies:** Keep minimal; only add if absolutely necessary

### Common Modification Scenarios

**Adding New Filters:**
1. Add parameter sliders in appropriate column
2. Apply filter in processing pipeline (between lines 57-60)
3. Add display column if showing intermediate results
4. Update README.md to document new feature

**Adjusting Parameter Ranges:**
- Check OpenCV documentation for valid ranges
- Consider educational value vs. practical utility
- Update slider min/max/default values accordingly

**Export Formats:**
- PNG export pattern: main.py:74-85 (uses PIL and BytesIO)
- SVG export pattern: main.py:88-92 (uses base64 encoding)
- Maintain download button UX consistency

### Critical Implementation Details

**Grayscale Handling (main.py:40-43):**
```python
if len(image_array.shape) == 3:
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
else:
    gray_image = image_array
```
- Always check image dimensions before conversion
- Handles already-grayscale images gracefully

**SVG Path Construction (main.py:14-21):**
- Iterates through all contours found by OpenCV
- Builds SVG path with M (move) and L (line) commands
- Closes paths with Z command
- Critical: SVG coordinates match pixel coordinates exactly

**Streamlit Download Pattern:**
- PNG: Uses `st.download_button` directly (main.py:80-85)
- SVG: Uses base64-encoded data URI with markdown (main.py:90-92)
- Different patterns due to Streamlit API design

## Dependencies & Environment

### Python Version
- Minimum: Python 3.7
- Recommended: Python 3.8+

### External Dependencies

**streamlit**
- Purpose: Web UI framework
- Usage: All UI components (file_uploader, slider, image, button, download_button)

**opencv-python-headless**
- Purpose: Computer vision algorithms
- Usage: bilateralFilter, Canny, findContours, cvtColor
- Note: Headless version used (no GUI dependencies)

**pillow (PIL)**
- Purpose: Image file I/O
- Usage: Image.open, Image.fromarray, save to BytesIO

**numpy**
- Purpose: Array operations
- Usage: np.array for image data conversion
- Note: Implicit dependency via OpenCV

### System Requirements
- RAM: 4GB minimum for processing larger images
- Storage: Minimal (application itself is <10KB)
- Browser: Modern browser for Streamlit interface

## Common Tasks & Examples

### Adding a New Image Processing Step

```python
# Example: Adding Gaussian blur after bilateral filter
# In main.py, around line 58:

bilateral_filtered = cv2.bilateralFilter(gray_image, d, sigma_color, sigma_space)

# Add new parameter slider in col1 or col2
with col1:
    blur_kernel = st.slider('Gaussian Kernel Size', 3, 11, 5, step=2)

# Apply Gaussian blur
gaussian_blurred = cv2.GaussianBlur(bilateral_filtered, (blur_kernel, blur_kernel), 0)

# Update Canny to use blurred image
edges = cv2.Canny(gaussian_blurred, low_threshold, high_threshold)

# Add display column to show gaussian_blurred result
```

### Modifying Parameter Ranges

```python
# Original
low_threshold = st.slider('Canny Low Threshold', 50, 150, 100)

# Modified with wider range
low_threshold = st.slider('Canny Low Threshold', 0, 200, 100)
```

### Adding New Export Format (e.g., JPEG)

```python
# Add in columns section around line 72
col3 = st.columns(3)  # Change from 2 to 3
with col3:
    if st.button('Save as JPEG'):
        edge_image = Image.fromarray(edges)
        buf = io.BytesIO()
        edge_image.save(buf, format='JPEG')
        byte_im = buf.getvalue()

        st.download_button(
            label="Download JPEG",
            data=byte_im,
            file_name="canny_edges.jpg",
            mime="image/jpeg"
        )
```

## Known Limitations & Considerations

1. **No Error Handling**
   - No try-except blocks for image loading or processing
   - Assumes valid image inputs
   - Consider adding error handling for production use

2. **No Image Size Limits**
   - Very large images may cause performance issues
   - No validation or downscaling for oversized inputs
   - Streamlit may have upload size limits (default 200MB)

3. **Single-threaded Processing**
   - No parallelization or async processing
   - Each parameter change triggers full reprocessing
   - Consider caching strategies for optimization

4. **SVG Quality**
   - SVG conversion is basic (not smoothed/optimized)
   - Many small contours may create large SVG files
   - No simplification or path optimization applied

5. **No State Persistence**
   - Parameters reset on page refresh
   - No session storage or configuration saving
   - Each upload starts fresh

## Future Enhancement Opportunities

If expanding this application, consider:

1. **Testing Infrastructure**
   - Unit tests for `edge_to_svg()` function
   - Integration tests for processing pipeline
   - Test fixtures with sample images

2. **Modularization**
   - Separate image processing functions into dedicated module
   - Create utility module for export functions
   - Consider class-based structure for scalability

3. **Advanced Features**
   - Batch processing multiple images
   - Parameter presets/profiles
   - Real-time performance metrics
   - Comparison mode (before/after slider)

4. **Error Handling & Validation**
   - File size limits
   - Format validation
   - Graceful degradation on processing errors
   - User feedback for failures

5. **Performance Optimization**
   - Streamlit caching decorators (@st.cache_data)
   - Image downscaling for preview
   - Lazy processing (only on demand)

## Troubleshooting Guide

### Common Issues

**Streamlit won't start:**
- Verify Python version (3.7+)
- Check dependencies installed: `pip list | grep streamlit`
- Try: `python -m streamlit run main.py`

**OpenCV errors:**
- Ensure opencv-python-headless is installed (not opencv-python)
- Check for version conflicts: `pip show opencv-python-headless`
- Reinstall: `pip uninstall opencv-python opencv-python-headless && pip install opencv-python-headless`

**Image upload fails:**
- Verify file format (PNG, JPG, JPEG only)
- Check file size (Streamlit default limit: 200MB)
- Try smaller test image

**SVG download doesn't work:**
- Check browser allows data URI downloads
- Verify SVG string is valid (inspect element)
- Try PNG export as alternative

**Parameters have no effect:**
- Ensure sliders are being moved (Streamlit may batch updates)
- Check browser console for JavaScript errors
- Refresh page and try again

## Contact & Resources

### Documentation
- **User Guide:** See README.md for detailed usage instructions
- **Streamlit Docs:** https://docs.streamlit.io
- **OpenCV Docs:** https://docs.opencv.org

### Repository Information
- **Git Repository:** wabisukecx/Binalize_Image
- **Development Branch Pattern:** claude/claude-md-*

---

**Last Updated:** 2025-12-07
**Document Version:** 1.0
**Codebase Status:** Stable, production-ready for educational use
