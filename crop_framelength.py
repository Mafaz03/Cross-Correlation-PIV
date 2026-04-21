import cv2

import argparse


parser = argparse.ArgumentParser(description="Process files.")

parser.add_argument("-v_pth",   "--video_path",         type=str,           help="video path")
parser.add_argument("-vs_pth",  "--video_save_path",    type=str,           help="video save path")
parser.add_argument("-f",  "--frames",    type=int,           help="how many frames")

args = parser.parse_args()


cap = cv2.VideoCapture(args.video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

out = cv2.VideoWriter(args.video_save_path, fourcc, fps, (width, height))

frame_count = 0
while cap.isOpened() and frame_count < args.frames:
    ret, frame = cap.read()
    
    if not ret:
        break
    
    out.write(frame)
    frame_count += 1

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Successfully clipped {frame_count} frames.")
