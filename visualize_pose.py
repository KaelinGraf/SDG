import json
import cv2
import numpy as np
import sys
import os

def visualize_poses(json_path, image_path):
    if not os.path.exists(json_path) or not os.path.exists(image_path):
        print("Error: JSON or Image file not found.")
        return

    with open(json_path, 'r') as f:
        data = json.load(f)
    
    img = cv2.imread(image_path)
    if img is None:
        print("Could not read image")
        return
        
    cam_K = np.array(data['camera']['cam_K']).reshape(3, 3)
    
    for obj in data['objects']:
        t = np.array(obj['pose']['cam_t_m2c'])
        
        # Project 3D translation origin to 2D image coordinates
        # t is already in camera frame (+Z forward)
        z = t[2]
        if z <= 0.01:
            continue
            
        u = int((t[0] / z) * cam_K[0, 0] + cam_K[0, 2])
        v = int((t[1] / z) * cam_K[1, 1] + cam_K[1, 2])
        
        # Draw a red dot at the origin of the object
        cv2.circle(img, (u, v), 5, (0, 0, 255), -1)
        # Draw the class name
        cv2.putText(img, obj['class'], (u + 10, v), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    output_path = image_path.replace(".png", "_visualized.png")
    cv2.imwrite(output_path, img)
    print(f"Successfully drew projected origins onto: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python visualize_pose.py <json_path> <rgb_image_path>")
    else:
        visualize_poses(sys.argv[1], sys.argv[2])
