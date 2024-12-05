import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_and_prepare_image(image_path):
    """Load and convert image to grayscale"""
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, gray

def detect_sift_features(gray_img):
    """Detect SIFT features in an image"""
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray_img, None)
    return keypoints, descriptors

def detect_surf_features(gray_img):
    """Detect SURF features in an image (if available in OpenCV build)"""
    try:
        surf = cv2.xfeatures2d.SURF_create(400)
        keypoints, descriptors = surf.detectAndCompute(gray_img, None)
        return keypoints, descriptors
    except:
        print("SURF is not available in this OpenCV build")
        return None, None

def detect_orb_features(gray_img):
    """Detect ORB features in an image"""
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray_img, None)
    return keypoints, descriptors

def detect_harris_corners(gray_img):
    """Detect Harris corners in an image"""
    dst = cv2.cornerHarris(gray_img, blockSize=2, ksize=3, k=0.04)
    # Dilate corner detections
    dst = cv2.dilate(dst, None)
    return dst

def match_features(desc1, desc2, method='bf'):
    """Match features between two images"""
    if method == 'bf':
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(desc1, desc2)
        matches = sorted(matches, key=lambda x: x.distance)
    elif method == 'flann':
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(desc1, desc2, k=2)
    return matches

def visualize_features(img, keypoints, title="Features"):
    """Visualize detected features"""
    img_with_features = cv2.drawKeypoints(
        img, 
        keypoints, 
        None, 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_with_features, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def visualize_matches(img1, kp1, img2, kp2, matches):
    """Visualize matched features between two images"""
    match_img = cv2.drawMatches(
        img1, kp1, 
        img2, kp2, 
        matches[:10], None, 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(15, 5))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title("Feature Matches")
    plt.axis('off')
    plt.show()

# Example usage:
if __name__ == "__main__":
    # Load images
    # img1, gray1 = load_and_prepare_image('pexels-asadphoto-240526.jpg')
    img1, gray1 = load_and_prepare_image('images.png')
    # img1, gray1 = load_and_prepare_image('pexels-mikebirdy-112460.jpg')
    img2, gray2 = load_and_prepare_image('pexels-mikebirdy-116675.jpg')
    
    # 1. SIFT Features
    kp1_sift, desc1_sift = detect_sift_features(gray1)
    visualize_features(img1, kp1_sift, "SIFT Features")
    
    # 2. ORB Features
    kp1_orb, desc1_orb = detect_orb_features(gray1)
    visualize_features(img1, kp1_orb, "ORB Features")
    
    # 3. Harris Corners
    corners = detect_harris_corners(gray1)
    plt.figure(figsize=(10, 8))
    plt.imshow(corners, cmap='gray')
    plt.title("Harris Corners")
    plt.axis('off')
    plt.show()
    
    # 4. Feature Matching Example
    kp2_sift, desc2_sift = detect_sift_features(gray2)
    matches = match_features(desc1_sift, desc2_sift, method='bf')
    visualize_matches(img1, kp1_sift, img2, kp2_sift, matches)