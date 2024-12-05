import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self, image_path):
        """Initialize with an image path."""
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise ValueError("Could not load image")
        # Convert BGR to RGB for matplotlib display
        self.original = cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB)

    def grayscale_conversion(self):
        """Convert RGB image to grayscale."""
        return cv2.cvtColor(self.original, cv2.COLOR_RGB2GRAY)

    def thresholding(self, gray_image, method='global'):
        """
        Apply thresholding to grayscale image.
        
        Args:
            gray_image: Grayscale input image
            method: 'global' or 'adaptive'
        """
        if method == 'global':
            # Otsu's thresholding
            _, binary = cv2.threshold(gray_image, 0, 255, 
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # Adaptive thresholding
            binary = cv2.adaptiveThreshold(gray_image, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        return binary

    def edge_detection(self, gray_image, method='canny'):
        """
        Detect edges in the image.
        
        Args:
            gray_image: Grayscale input image
            method: 'canny' or 'sobel'
        """
        if method == 'canny':
            return cv2.Canny(gray_image, 100, 200)
        else:
            # Sobel edge detection
            sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
            return np.uint8(np.sqrt(sobel_x**2 + sobel_y**2))

    def image_transformations(self):
        """Apply various geometric transformations."""
        height, width = self.original.shape[:2]
        
        # Rotation
        matrix = cv2.getRotationMatrix2D((width/2, height/2), 45, 1.0)
        rotated = cv2.warpAffine(self.original, matrix, (width, height))
        
        # Scaling
        scaled = cv2.resize(self.original, None, fx=0.5, fy=0.5)
        
        # Cropping
        cropped = self.original[height//4:3*height//4, 
                              width//4:3*width//4]
        
        return rotated, scaled, cropped

    def filtering(self, gray_image):
        """Apply different filters to the image."""
        # Gaussian filter
        gaussian = cv2.GaussianBlur(gray_image, (5, 5), 0)
        
        # Median filter
        median = cv2.medianBlur(gray_image, 5)
        
        # Sharpening filter
        kernel = np.array([[-1,-1,-1],
                         [-1, 9,-1],
                         [-1,-1,-1]])
        sharpened = cv2.filter2D(gray_image, -1, kernel)
        
        return gaussian, median, sharpened

    def morphological_operations(self, binary_image):
        """Apply morphological operations."""
        kernel = np.ones((5,5), np.uint8)
        
        # Erosion
        erosion = cv2.erode(binary_image, kernel, iterations=1)
        
        # Dilation
        dilation = cv2.dilate(binary_image, kernel, iterations=1)
        
        return erosion, dilation


    def histogram_equalization(self, image):
        """
        Apply histogram equalization to enhance contrast.
        Works on both grayscale and color images.
        """
        if len(image.shape) == 2:  # Grayscale
            return cv2.equalizeHist(image)
        else:  # Color image
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply histogram equalization to L channel
            l_eq = cv2.equalizeHist(l)
            
            # Merge channels and convert back to RGB
            lab_eq = cv2.merge([l_eq, a, b])
            return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
    
    def color_space_conversions(self):
        """
        Convert image to different color spaces.
        Returns HSV, LAB, and YCrCb representations.
        """
        hsv = cv2.cvtColor(self.original, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(self.original, cv2.COLOR_RGB2LAB)
        ycrcb = cv2.cvtColor(self.original, cv2.COLOR_RGB2YCrCb)
        return hsv, lab, ycrcb
    
    def add_noise(self, image, noise_type='gaussian'):
        """
        Add different types of noise to the image.
        
        Args:
            image: Input image
            noise_type: 'gaussian' or 'salt_pepper'
        """
        if noise_type == 'gaussian':
            row, col, ch = image.shape
            mean = 0
            sigma = 25
            gauss = np.random.normal(mean, sigma, (row, col, ch))
            noisy = image + gauss
            return np.clip(noisy, 0, 255).astype(np.uint8)
        else:  # salt and pepper
            row, col, ch = image.shape
            s_vs_p = 0.5
            amount = 0.004
            noisy = np.copy(image)
            
            # Salt mode
            num_salt = np.ceil(amount * image.size * s_vs_p)
            coords = [np.random.randint(0, i - 1, int(num_salt))
                     for i in image.shape]
            noisy[coords[0], coords[1], :] = 255

            # Pepper mode
            num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
            coords = [np.random.randint(0, i - 1, int(num_pepper))
                     for i in image.shape]
            noisy[coords[0], coords[1], :] = 0
            return noisy
    
    def denoising(self, image, method='non_local_means'):
        """
        Apply different denoising methods.
        
        Args:
            image: Input image
            method: 'non_local_means' or 'bilateral'
        """
        if method == 'non_local_means':
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:  # bilateral
            return cv2.bilateralFilter(image, 9, 75, 75)
    
    def image_blending(self, second_image_path, alpha=0.5):
        """
        Blend two images together.
        
        Args:
            second_image_path: Path to second image
            alpha: Blending factor (0.0 - 1.0)
        """
        img2 = cv2.imread(second_image_path)
        if img2 is None:
            raise ValueError("Could not load second image")
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Resize second image to match first image
        img2 = cv2.resize(img2, (self.original.shape[1], self.original.shape[0]))
        
        return cv2.addWeighted(self.original, alpha, img2, 1-alpha, 0)
    
    def perspective_transform(self, points_src, points_dst):
        """
        Apply perspective transformation.
        
        Args:
            points_src: Four source points
            points_dst: Four destination points
        """
        matrix = cv2.getPerspectiveTransform(points_src, points_dst)
        height, width = self.original.shape[:2]
        return cv2.warpPerspective(self.original, matrix, (width, height))
    
    def color_segmentation(self, lower_bound, upper_bound):
        """
        Segment image based on color range in HSV space.
        
        Args:
            lower_bound: Lower HSV values [H, S, V]
            upper_bound: Upper HSV values [H, S, V]
        """
        hsv = cv2.cvtColor(self.original, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array(lower_bound), np.array(upper_bound))
        return cv2.bitwise_and(self.original, self.original, mask=mask)
    
    def template_matching(self, template_path, method=cv2.TM_CCOEFF_NORMED):
        """
        Find a template in the image.
        
        Args:
            template_path: Path to template image
            method: Template matching method
        """
        template = cv2.imread(template_path)
        if template is None:
            raise ValueError("Could not load template image")
        template = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
        
        result = cv2.matchTemplate(self.original, template, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Get the coordinates for drawing rectangle
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
            
        h, w = template.shape[:2]
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        # Draw rectangle on copy of original image
        result_img = self.original.copy()
        cv2.rectangle(result_img, top_left, bottom_right, (0, 255, 0), 2)
        return result_img

    def display_results(self):
        """Display all image processing results."""
        # Convert to grayscale
        gray = self.grayscale_conversion()
        
        # Thresholding
        global_thresh = self.thresholding(gray, 'global')
        adaptive_thresh = self.thresholding(gray, 'adaptive')
        
        # Edge detection
        canny_edges = self.edge_detection(gray, 'canny')
        sobel_edges = self.edge_detection(gray, 'sobel')
        
        # Transformations
        rotated, scaled, cropped = self.image_transformations()
        
        # Filtering
        gaussian, median, sharpened = self.filtering(gray)
        
        # Morphological operations
        erosion, dilation = self.morphological_operations(global_thresh)
        
        # Create figure with subplots
        fig, axes = plt.subplots(4, 4, figsize=(15, 15))
        fig.suptitle('Image Processing Techniques', fontsize=16)
        
        # Original and grayscale
        axes[0,0].imshow(self.original)
        axes[0,0].set_title('Original')
        axes[0,1].imshow(gray, cmap='gray')
        axes[0,1].set_title('Grayscale')
        
        # Thresholding
        axes[0,2].imshow(global_thresh, cmap='gray')
        axes[0,2].set_title('Global Thresholding')
        axes[0,3].imshow(adaptive_thresh, cmap='gray')
        axes[0,3].set_title('Adaptive Thresholding')
        
        # Edge detection
        axes[1,0].imshow(canny_edges, cmap='gray')
        axes[1,0].set_title('Canny Edges')
        axes[1,1].imshow(sobel_edges, cmap='gray')
        axes[1,1].set_title('Sobel Edges')
        
        # Transformations
        axes[1,2].imshow(rotated)
        axes[1,2].set_title('Rotated')
        axes[1,3].imshow(scaled)
        axes[1,3].set_title('Scaled')
        
        # More transformations and filtering
        axes[2,0].imshow(cropped)
        axes[2,0].set_title('Cropped')
        axes[2,1].imshow(gaussian, cmap='gray')
        axes[2,1].set_title('Gaussian Filter')
        axes[2,2].imshow(median, cmap='gray')
        axes[2,2].set_title('Median Filter')
        axes[2,3].imshow(sharpened, cmap='gray')
        axes[2,3].set_title('Sharpened')
        
        # Morphological operations
        axes[3,0].imshow(erosion, cmap='gray')
        axes[3,0].set_title('Erosion')
        axes[3,1].imshow(dilation, cmap='gray')
        axes[3,1].set_title('Dilation')
        
        # Remove axis ticks
        for row in axes:
            for ax in row:
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()

    def display_enhanced_results(self):
        """Display results of enhanced processing methods."""
        # Create subplots
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        fig.suptitle('Enhanced Image Processing Techniques', fontsize=10)
        
        # Original image
        axes[0,0].imshow(self.original)
        axes[0,0].set_title('Original')
        
        # Histogram equalization
        gray = cv2.cvtColor(self.original, cv2.COLOR_RGB2GRAY)
        hist_eq = self.histogram_equalization(self.original)
        axes[0,1].imshow(hist_eq)
        axes[0,1].set_title('Histogram Equalization')
        
        # Color spaces
        hsv, lab, ycrcb = self.color_space_conversions()
        axes[0,2].imshow(hsv)
        axes[0,2].set_title('HSV Color Space')
        axes[0,3].imshow(lab)
        axes[0,3].set_title('LAB Color Space')
        
        # Noise addition
        gaussian_noise = self.add_noise(self.original, 'gaussian')
        sp_noise = self.add_noise(self.original, 'salt_pepper')
        axes[1,0].imshow(gaussian_noise)
        axes[1,0].set_title('Gaussian Noise')
        axes[1,1].imshow(sp_noise)
        axes[1,1].set_title('Salt & Pepper Noise')
        
        # Denoising
        denoised_nlm = self.denoising(gaussian_noise, 'non_local_means')
        denoised_bilateral = self.denoising(gaussian_noise, 'bilateral')
        axes[1,2].imshow(denoised_nlm)
        axes[1,2].set_title('Non-Local Means Denoising')
        axes[1,3].imshow(denoised_bilateral)
        axes[1,3].set_title('Bilateral Denoising')
        
        # Remove axis ticks
        for row in axes:
            for ax in row:
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()

# Usage example
if __name__ == "__main__":
    processor = ImageProcessor("pexels-asadphoto-240526.jpg")
    processor.display_results()
    processor.display_enhanced_results()