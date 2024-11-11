import cv2 
import os

#function to extract the frames of the video
def extract_frames(video_path, output_path):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load the video
    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    #read through the videoto make into frames
    while video.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if no more frames are available

        # Save frame as an image file
        frame_filename = os.path.join(output_dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_filename, frame)
        
        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Extracted {frame_count} frames from the video.")
    
# function to find the similarity between the different frames
# to implement the shot boundary detection
def find_similarity_between_frames(frames_path):
    
    
    
    
if __name__ == "__main__":
    videopath = ""
    output_frames = "/extracted_frames"
    
    