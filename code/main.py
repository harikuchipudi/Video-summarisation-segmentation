import cv2
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
import faiss
from PIL import Image
import matplotlib.pyplot as plt

import easyocr
import yolov5
from PIL import Image
from torchvision.models import vit_b_16



# YOLOv5 setup
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
reader = easyocr.Reader(['en'])

# ResNet50 and ViT setup
resnet_model = resnet50(pretrained=True)
resnet_model.eval()
vit_model = vit_b_16(pretrained=True)
vit_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# Function to extract the frames of the video
def extract_frames(video_path, output_path):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load the video
    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    # Read through the video to extract frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if no more frames are available

        # Save frame as an image file
        frame_filename = os.path.join(output_path, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)

        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Extracted {frame_count} frames from the video.")
    return frame_count

# Function to find the similarity between different frames
def histogram_similarity(frameA, frameB):
    histA = cv2.calcHist([frameA], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    histB = cv2.calcHist([frameB], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(histA, histA)
    cv2.normalize(histB, histB)
    similarity = cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)
    return similarity

def ssim_similarity(frameA, frameB):
    grayA = cv2.cvtColor(frameA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(frameB, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(grayA, grayB, full=True)
    return score

# Shot boundary detection
def segment_video(video_path, hist_threshold=0.85, ssim_threshold=0.5, output_dir="segments"):
    # Set up video capture and create output directory
    cap = cv2.VideoCapture(video_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    shot_boundaries = []
    prev_frame = None
    frame_count = 0
    shot_count = 0

    # Video reading loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if prev_frame is not None:
            # Calculate histogram and SSIM similarity
            hist_sim = histogram_similarity(prev_frame, frame)
            ssim_sim = ssim_similarity(prev_frame, frame)

            # Check if either similarity metric indicates a shot change
            if hist_sim < hist_threshold or ssim_sim < ssim_threshold:
                shot_boundaries.append(frame_count)

                # Save frame as representative image for this shot
                cv2.imwrite(f"{output_dir}/shot_{shot_count}.jpg", prev_frame)
                shot_count += 1
                print(f"Shot boundary detected at frame {frame_count}")

        # Update previous frame and increment frame count
        prev_frame = frame
        frame_count += 1

    # Release video and close windows
    cap.release()
    print("Shot boundaries detected at frames:", shot_boundaries)
    print(f"Segments and summary images saved in '{output_dir}' directory.")
    return shot_boundaries

# Extract features using a pre-trained ResNet
def extract_features(frames_dir, key_frames):
    model = resnet50(pretrained=True)
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    features = []

    if not key_frames:
        return []

    for frame_idx in key_frames:
        frame_path = os.path.join(frames_dir, f'frame_{frame_idx:04d}.jpg')
        if not os.path.exists(frame_path):
            print(f"Warning: Frame {frame_idx} not found.")
            continue

        image = Image.open(frame_path).convert("RGB")
        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            feature = model(input_tensor)
            features.append(feature.squeeze().numpy())
    return features

def extract_frame_features(frame):
    frame_features = {}
    # YOLOv5
    results = model_yolo(frame)
    detected_objects = results.xyxy[0].cpu().numpy()  # Bounding boxes, confidence, and class
    frame_features['detected_objects'] = detected_objects

    # EasyOCR
    text_results = reader.readtext(frame)
    frame_features['detected_texts'] = text_results

    # ResNet50 and ViT feature extraction
    pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_frame).unsqueeze(0)
    with torch.no_grad():
        resnet_features = resnet_model(input_tensor).squeeze().numpy()
        vit_features = vit_model(input_tensor).squeeze().numpy()
    frame_features['resnet_features'] = resnet_features
    frame_features['vit_features'] = vit_features

    return frame_features

def visualize_results(frame, features, frame_idx):
    display_frame = frame.copy()
    for obj in features['detected_objects']:
        x1, y1, x2, y2, confidence, cls = obj
        label = f"Class: {int(cls)}, Conf: {confidence:.2f}"
        color = (0, 255, 0)
        cv2.rectangle(display_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(display_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    for text_result in features['detected_texts']:
        position, text, _ = text_result
        (x_min, y_min), (x_max, y_max) = position[0], position[2]
        cv2.rectangle(display_frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
        cv2.putText(display_frame, text, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    plt.figure(figsize=(10, 6))
    plt.title(f"Frame {frame_idx} Visualization")
    plt.imshow(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

def process_and_visualize_video(video_path, output_text_file="detection_results.txt", output_image_dir="visualizations"):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    if not os.path.exists(output_image_dir):
        os.makedirs(output_image_dir)
    
    # Open the text file to store results
    with open(output_text_file, 'w') as result_file:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process and visualize only every 10th frame
            if frame_count % 10 == 0:
                # Extract features
                features = extract_frame_features(frame)
                
                # Write detection results to text file
                result_file.write(f"Frame {frame_count}:\n")
                result_file.write(f"  Detected Objects: {features['detected_objects']}\n")
                result_file.write(f"  Detected Texts: {features['detected_texts']}\n")
                result_file.write("\n")

                # Save frame visualization
                visualize_results(frame, features, frame_count)
                visualization_path = os.path.join(output_image_dir, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(visualization_path, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            frame_count += 1

    cap.release()
    print(f"Results saved in {output_text_file} and visualizations in {output_image_dir} directory.")
 

# Content-based indexing using FAISS
def create_faiss_index(features):
    feature_dim = features[0].shape[0]
    index = faiss.IndexFlatL2(feature_dim)
    index.add(np.array(features, dtype=np.float32))
    return index

def search_similar_features(index, query_feature, top_k=5):
    query_feature = np.expand_dims(query_feature, axis=0).astype(np.float32)
    distances, indices = index.search(query_feature, top_k)
    return distances, indices

# just for testing
def compute_histogram_similarity(frameA, frameB):
    # Calculate histograms
    histA = cv2.calcHist([frameA], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    histB = cv2.calcHist([frameB], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    
    # Normalize histograms
    cv2.normalize(histA, histA)
    cv2.normalize(histB, histB)
    
    # Compare histograms
    similarity = cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)
    return similarity

def compute_ssim(frameA, frameB):
    # Convert frames to grayscale
    grayA = cv2.cvtColor(frameA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(frameB, cv2.COLOR_BGR2GRAY)
    
    # Compute SSIM
    score, _ = ssim(grayA, grayB, full=True)
    return score

def compare_frames(frames):
    for i in range(len(frames) - 1):
        # Load consecutive frames
        frameA = cv2.imread(frames[i])
        frameB = cv2.imread(frames[i + 1])
        
        # Compute similarities
        hist_similarity = compute_histogram_similarity(frameA, frameB)
        ssim_similarity = compute_ssim(frameA, frameB)
        
        # Print results
        print(f"Comparison between Frame {i} and Frame {i + 1}:")
        print(f"  Histogram Similarity: {hist_similarity:.4f}")
        print(f"  SSIM: {ssim_similarity:.4f}")
        print()
        
def plot_frame_and_histogram(frame_path):
    # Load the frame
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Error: Cannot load frame {frame_path}")
        return

    # Convert the frame to RGB for matplotlib
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Calculate histograms for each color channel
    color = ('b', 'g', 'r')  # Blue, Green, Red in OpenCV
    histograms = {}
    for i, col in enumerate(color):
        hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
        histograms[col] = hist

    # Plot the frame and histograms
    plt.figure(figsize=(10, 5))
    
    # Display the frame
    plt.subplot(1, 2, 1)
    plt.imshow(frame_rgb)
    plt.title(f"Frame: {frame_path.split('/')[-1]}")
    plt.axis("off")
    
    # Plot the histograms
    plt.subplot(1, 2, 2)
    for col, hist in histograms.items():
        plt.plot(hist, color=col)
    plt.title("Color Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend(["Blue", "Green", "Red"])

    # Show the plots
    plt.tight_layout()
    plt.show()



# Main script
if __name__ == "__main__":
    video_path = "samplevideo.mp4"
    output_frames_dir = "./extracted_frames"
    output_segments_dir = "./segments"
    process_and_visualize_video(video_path)
    
    # frames = ["/Users/hari/Documents/GitHub/Video-summarisation-segmentation/code/extracted_frames/frame_0138.jpg", 
    #           "/Users/hari/Documents/GitHub/Video-summarisation-segmentation/code/extracted_frames/frame_0139.jpg",
    #           "/Users/hari/Documents/GitHub/Video-summarisation-segmentation/code/extracted_frames/frame_0142.jpg",
    #           "/Users/hari/Documents/GitHub/Video-summarisation-segmentation/code/extracted_frames/frame_0143.jpg",
    #           "/Users/hari/Documents/GitHub/Video-summarisation-segmentation/code/extracted_frames/frame_0144.jpg",
    #           "/Users/hari/Documents/GitHub/Video-summarisation-segmentation/code/extracted_frames/frame_0145.jpg",
    #           "/Users/hari/Documents/GitHub/Video-summarisation-segmentation/code/extracted_frames/frame_0146.jpg"]
    
    # compare_frames(frames)
    
    # for frame_path in frames:
    #     plot_frame_and_histogram(frame_path)
        
    
        

    # # Step 1: Frame extraction
    # frame_count = extract_frames(video_path, output_frames_dir)

    # # Step 2: Shot boundary detection
    # shot_boundaries = segment_video(video_path, output_dir=output_segments_dir)

    # # Step 3: Key frame selection (simplified: using shot boundaries)
    # key_frames = shot_boundaries

    # # Step 4: Feature extraction
    # if key_frames:
    #     features = extract_features(output_frames_dir, key_frames)
    #     print(f"Extracted features for {len(features)} key frames.")
    # else:
    #     print("No key frames detected. Exiting.")
    #     exit()

    # # Step 5: Content-based indexing
    # faiss_index = create_faiss_index(features)
    # print("Features indexed using FAISS.")

    # # Step 6: Query example
    # query_feature = features[0]
    # distances, indices = search_similar_features(faiss_index, query_feature)
    # print("Top matches for the query frame:", indices)

