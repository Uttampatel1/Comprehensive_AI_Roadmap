import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from scipy.spatial import distance
import tensorflow as tf
from pathlib import Path
import matplotlib.pyplot as plt

class ImageFeatureExtractor:
    def __init__(self):
        self.sift = cv2.SIFT_create()
        self.orb = cv2.ORB_create()
        self.model = None
        
    def visualize_result(self, image, title="Result", save_path=None):
        """Helper function to visualize and save results"""
        plt.figure(figsize=(12, 8))
        if len(image.shape) == 3:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        
    def extract_bow_features(self, image_path, vocabulary_size=1000):
        """
        Bag of Visual Words feature extraction with visualization
        """
        img = cv2.imread(str(image_path))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Extract SIFT features
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        if descriptors is None:
            return None
            
        # Create vocabulary if not exists
        if self.model is None:
            self.model = KMeans(n_clusters=vocabulary_size, random_state=42)
            self.model.fit(descriptors)
            
        # Get histogram of features
        histogram = np.zeros(vocabulary_size)
        words = self.model.predict(descriptors)
        for word in words:
            histogram[word] += 1
            
        # Visualize keypoints
        img_keypoints = cv2.drawKeypoints(img, keypoints, None, 
                                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        self.visualize_result(img_keypoints, "Bag of Visual Words Features", 
                            "outputs/bow_features.png")
            
        return histogram / len(keypoints) if keypoints else None

    def detect_object_realtime(self, template_path, threshold=0.8, save_frames=True):
        """
        Real-time object detection with frame saving
        """
        template = cv2.imread(template_path, 0)
        cap = cv2.VideoCapture(0)
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            
            loc = np.where(result >= threshold)
            detected_frame = frame.copy()
            for pt in zip(*loc[::-1]):
                cv2.rectangle(detected_frame, pt, 
                            (pt[0] + template.shape[1], pt[1] + template.shape[0]), 
                            (0, 255, 0), 2)
            
            if save_frames and frame_count % 30 == 0:  # Save every 30th frame
                self.visualize_result(detected_frame, 
                                    f"Object Detection Frame {frame_count}", 
                                    f"outputs/detection_frame_{frame_count}.png")
            
            cv2.imshow('Object Detection', detected_frame)
            frame_count += 1
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()

    def facial_landmark_detection(self, image_path):
        """
        Facial landmark detection with visualization
        """
        import dlib
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = detector(gray)
        landmarks = []
        
        for face in faces:
            shape = predictor(gray, face)
            points = np.zeros((68, 2), dtype=int)
            
            for i in range(68):
                points[i] = (shape.part(i).x, shape.part(i).y)
                cv2.circle(img, (points[i][0], points[i][1]), 2, (0, 255, 0), -1)
                
            landmarks.append(points)
            
        # Save the visualization
        self.visualize_result(img, "Facial Landmarks", "outputs/facial_landmarks.png")
        return img, landmarks

    def image_retrieval_system(self, query_image, image_database):
        """
        Content-based image retrieval with visualization
        """
        def compute_features(img):
            keypoints, descriptors = self.sift.detectAndCompute(img, None)
            return keypoints, descriptors
            
        query_img = cv2.imread(query_image, 0)
        query_kp, query_desc = compute_features(query_img)
        
        results = []
        for img_path in image_database:
            db_img = cv2.imread(str(img_path), 0)
            db_kp, db_desc = compute_features(db_img)
            
            if db_desc is not None and query_desc is not None:
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                
                flann = cv2.FlannBasedMatcher(index_params, search_params)
                matches = flann.knnMatch(query_desc, db_desc, k=2)
                
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
                
                # Visualize matches
                match_img = cv2.drawMatches(query_img, query_kp, db_img, db_kp, 
                                          good_matches[:10], None, flags=2)
                self.visualize_result(match_img, f"Matches with {img_path}", 
                                    f"outputs/matches_{Path(img_path).stem}.png")
                        
                results.append((img_path, len(good_matches)))
                
        return sorted(results, key=lambda x: x[1], reverse=True)

    def panorama_stitching(self, images):
        """
        Image stitching with visualization
        """
        stitcher = cv2.Stitcher_create()
        status, panorama = stitcher.stitch(images)
        
        if status == cv2.Stitcher_OK:
            self.visualize_result(panorama, "Panorama Stitching", 
                                "outputs/panorama.png")
            return panorama
        else:
            print("Stitching failed!")
            return None

    def text_detection(self, image_path):
        """
        Text detection with visualization
        """
        img = cv2.imread(image_path)
        orig = img.copy()
        (H, W) = img.shape[:2]

        (newW, newH) = (320, 320)
        rW = W / float(newW)
        rH = H / float(newH)

        img = cv2.resize(img, (newW, newH))
        (H, W) = img.shape[:2]

        layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
            "feature_fusion/concat_3"
        ]

        net = cv2.dnn.readNet("frozen_east_text_detection.pb")
        blob = cv2.dnn.blobFromImage(img, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
        net.setInput(blob)
        (scores, geometry) = net.forward(layerNames)

        # Visualize confidence scores
        confidence_map = scores[0, 0]
        self.visualize_result(confidence_map, "Text Detection Confidence Map", 
                            "outputs/text_detection_confidence.png")

        return scores, geometry, orig, rW, rH

def main():
    # Create output directory if it doesn't exist
    Path("outputs").mkdir(exist_ok=True)
    
    extractor = ImageFeatureExtractor()
    
    # Bag of Visual Words
    print("Processing Bag of Visual Words...")
    histogram = extractor.extract_bow_features('feature_inputs_images/fa1.png')
    
    # Real-time object detection
    print("Processing Object Detection...")
    # extractor.detect_object_realtime('feature_inputs_images/fa2.png')
    
    # Facial landmark detection
    print("Processing Facial Landmarks...")
    result_img, landmarks = extractor.facial_landmark_detection('feature_inputs_images/fa3.png')
    
    # Panorama stitching
    print("Processing Panorama Stitching...")
    images = [cv2.imread(f) for f in ['feature_inputs_images/fa42.png', 
                                     'feature_inputs_images/fa41.png', 
                                     'feature_inputs_images/fa43.png']]
    panorama = extractor.panorama_stitching(images)
    
    # Text detection
    print("Processing Text Detection...")
    scores, geometry, orig, rW, rH = extractor.text_detection('feature_inputs_images/fa4.png')
    
    print("Processing complete! Check the 'outputs' directory for results.")

if __name__ == "__main__":
    main()