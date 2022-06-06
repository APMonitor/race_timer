import os
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# For video input, switch to 0 (number) for webcam input
f = '1.mp4'
# f = 0 # for webcam input
cap = cv2.VideoCapture(f)

# Output video
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# Write video.avi
out = cv2.VideoWriter('form_'+f, \
                     cv2.VideoWriter_fourcc(*'XVID'), \
                     30, (w,h))

out2 = cv2.VideoWriter('form_cut_'+f, \
                       cv2.VideoWriter_fourcc(*'XVID'), \
                       30, (w,h))

BG_COLOR = (192, 192, 192) # gray


with mp_pose.Pose(
    min_detection_confidence=0.5,
    enable_segmentation=True,
    min_tracking_confidence=0.5) as pose:
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      # If loading a video, use 'break' instead of 'continue'.
      break

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    # Draw the pose annotation on the image.
    image2 = image.copy()
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    out.write(image)

    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    image2.flags.writeable = True
    image2 = cv2.cvtColor(image2, cv2.COLOR_RGB2BGR)
    try:
      condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
      bg_image = np.zeros(image2.shape, dtype=np.uint8)
      bg_image[:] = BG_COLOR
      image2 = np.where(condition, image2, bg_image)
      # Draw pose landmarks on the image.
      mp_drawing.draw_landmarks(
        image2,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
    except:
      continue
    out2.write(image2)
    
    # Plot pose world landmarks.
    #mp_drawing.plot_landmarks(
    #    results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)
    #plt.savefig('f_'+str(icount)+'.png')
    #icount +=1
    
cap.release()
out.release()
out2.release()
cv2.destroyAllWindows()
