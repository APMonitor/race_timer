# Analysis of 240 fps video taken with an iPhone 11
#  for comparison with Motion Capture results using reflective balls
#  also captured at 240 fps 

import os
import glob
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import time
from mediapipe.framework.formats import landmark_pb2
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

from typing import Tuple, Optional, List

# get file list to process
files = glob.glob('*.mov') # change to '*.mp4' for another file extension
#files = ['j2.mov']        # test with one file

try:
  os.mkdir('results') # create 'results' directory if it doesn't exist
except:
  continue

WHITE_COLOR = (224, 224, 224)
RED_COLOR = (0, 0, 255)
BLACK_COLOR = (0, 0, 0)
_VISIBILITY_THRESHOLD = 0.5

def _normalize_color(color):
  return tuple(v / 255. for v in color)

# modified function from mediapipe package to not show plot
def plot_landmarks(landmark_list: landmark_pb2.NormalizedLandmarkList,
                   connections: Optional[List[Tuple[int, int]]] = None,
                   elevation: int = 10,
                   azimuth: int = 10):
  """Plot the landmarks and the connections in matplotlib 3d.

  Args:
    landmark_list: A normalized landmark list proto message to be plotted.
    connections: A list of landmark index tuples that specifies how landmarks to
      be connected.
    elevation: The elevation from which to view the plot.
    azimuth: the azimuth angle to rotate the plot.
  Raises:
    ValueError: If any connetions contain invalid landmark index.
  """
  thk = 2 # thickness
  if not landmark_list:
    return
  plt.figure(figsize=(10, 10))
  ax = plt.axes(projection='3d')
  ax.view_init(elev=elevation, azim=azimuth)
  plotted_landmarks = {}
  for idx, landmark in enumerate(landmark_list.landmark):
    if ((landmark.HasField('visibility') and
         landmark.visibility < _VISIBILITY_THRESHOLD) or
        (landmark.HasField('presence') and
         landmark.presence < _PRESENCE_THRESHOLD)):
      continue
    ax.scatter3D(
        xs=[-landmark.z],
        ys=[landmark.x],
        zs=[-landmark.y],
        color=_normalize_color(RED_COLOR),
        linewidth=thk)
    plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)
  if connections:
    num_landmarks = len(landmark_list.landmark)
    # Draws the connections if the start and end landmarks are both visible.
    for connection in connections:
      start_idx = connection[0]
      end_idx = connection[1]
      if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
        raise ValueError(f'Landmark index is out of range. Invalid connection '
                         f'from landmark #{start_idx} to landmark #{end_idx}.')
      if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
        landmark_pair = [
            plotted_landmarks[start_idx], plotted_landmarks[end_idx]
        ]
        ax.plot3D(
            xs=[landmark_pair[0][0], landmark_pair[1][0]],
            ys=[landmark_pair[0][1], landmark_pair[1][1]],
            zs=[landmark_pair[0][2], landmark_pair[1][2]],
            color=_normalize_color(BLACK_COLOR),
            linewidth=thk)

# cycle through video files
for f in files:
    start = time.time()
    print('-'*30)
    print('Processing ',f)
    # For video input, switch to 0 (number) for webcam input
    # f = '3.mp4'
    # f = 0 # for webcam input
    cap = cv2.VideoCapture(f)

    # Output videos
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))    
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Write video.avi
    out = cv2.VideoWriter('./results/F_'+f, \
                         cv2.VideoWriter_fourcc(*'XVID'), \
                         240, (w,h))

    out2 = cv2.VideoWriter('./results/FC_'+f, \
                           cv2.VideoWriter_fourcc(*'XVID'), \
                           60, (w,h))

    # plot images
    out3 = cv2.VideoWriter('./results/WP_'+f, \
                           cv2.VideoWriter_fourcc(*'XVID'), \
                           60, (3000,3000))

    # combined
    out4 = cv2.VideoWriter('./results/C_'+f, \
                           cv2.VideoWriter_fourcc(*'XVID'), \
                           10, (w,h))


    BG_COLOR = (192, 192, 192) # gray

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        enable_segmentation=True,
        min_tracking_confidence=0.5) as pose:
      ic = 0
      # read each frame from video
      while cap.isOpened():
        success, image = cap.read()
        ic += 1
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
          out2.write(image2)

          # Plot pose world landmarks.
          if ic<=2400: #True: #ic<=3:
            if ic%4==0:
              #mp_drawing.plot_landmarks
              plot_landmarks(results.pose_world_landmarks, \
                             mp_pose.POSE_CONNECTIONS,azimuth=(ic/10.0)%360)
              ax = plt.gca()
              ax.set_xticks([-0.8,-0.4,0,0.4,0.8])
              ax.set_yticks([-0.8,-0.4,0,0.4,0.8])
              ax.set_zticks([-0.6,-0.4,-0.2,0,0.2,0.4,0.6])

              ax.axes.set_xlim3d(left=-1.0, right=1.0) 
              ax.axes.set_ylim3d(bottom=-1.0, top=1.0) 
              ax.axes.set_zlim3d(bottom=-0.7, top=0.7)
          
              #plt.draw()
              plt.tight_layout()
              plt.savefig('tmp.png',transparent=True,dpi=300)
              plt.close()
          
              img = cv2.imread('tmp.png')
              out3.write(img)

              # add transparent plot to video
              img = cv2.imread('tmp.png',cv2.IMREAD_UNCHANGED)
              width=img.shape[1]; height=img.shape[0]
              hscaled=int(image2.shape[0]*0.7); #px
              scale=hscaled/height
              wscaled=int(width*scale)
              img = cv2.resize(img, (wscaled,hscaled),\
                               interpolation = cv2.INTER_AREA)
              alpha = img[:,:,3] / 255.0
              beta = []
              for i in range(3):
                beta.append(img[:,:,i] * (alpha))          
            
              # insert plot with transparency
              yloc = 0
              xloc = 0
              img1 = image2[yloc:yloc+hscaled,xloc:xloc+wscaled,:]
              for c in range(0,3):
                color = img1[:,:,c] * (1.0-alpha)
                img1[:,:,c] = color + beta[c]
              image2[yloc:yloc+hscaled,xloc:xloc+wscaled,:] = img1
              out4.write(image2)
            
        except:
          continue
        
    cap.release()
    out.release()
    out2.release()
    out3.release()
    out4.release()
    cv2.destroyAllWindows()
    
    print(' Processing Time: ',(time.time()-start)/60.0, ' min')
