import cv2 
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim

#function to extract the frames of the video
def extract_frames(video_path, output_path):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load the video
    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    #read through the videoto make into frames
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
    
# function to find the similarity between the different frames
# to implement the shot boundary detection
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

    
    
    
if __name__ == "__main__":
    video_path = "samplevideo.mp4"
    output_frames = "./extracted_frames"
    ouput_segments_dir = "./segements"
    
    extract_frames(video_path, output_frames)
    segment_video(video_path, output_dir="output_segments_dir")
    
    
    