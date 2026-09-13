import cv2
import numpy as np
from ant_env_grid import * 

#given trajectory
#open up the grid
#get the image 
#load into a video

env = environment()

def get_stream(trajectory,videofilename):
    #create video
    first_frame = env.getObservation()
    # Convert to NumPy array if necessary
    first_frame = np.asarray(first_frame)
    if first_frame.dtype != np.uint8:
        first_frame = first_frame.astype(np.uint8)
    height, width = first_frame.shape[:2]
    x1,y1=trajectory[i]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(
        videofilename,
        fourcc,
        30,
        (width, height)
    )
    video.write(first_frame)
    for i in range(1,len(trajectory)):
        x2,y2=trajectory[i]
        velx = x2 - x1
        vely = y2 - y1
        env.moveAgent(velx,vely) #move agent by velocity
        view_at_point=env.getObservation() #get the ant vision
        view_at_point = np.asarray(view_at_point)
        # Make sure OpenCV can write the frame
        if view_at_point.dtype != np.uint8:
            view_at_point = view_at_point.astype(np.uint8)

        # If the observation is grayscale, convert to BGR
        if len(view_at_point.shape) == 2:
            view_at_point = cv2.cvtColor(
                view_at_point,
                cv2.COLOR_GRAY2BGR
            )

        # Resize if necessary
        if (view_at_point.shape[1], view_at_point.shape[0]) != (width, height):
            view_at_point = cv2.resize(
                view_at_point,
                (width, height)
            )

        # Save frame to video
        video.write(view_at_point)
        #save to video stream
    #close video
    video.release()


def get_numpy(trajectory,filename):
    #create video
    first_frame = env.getObservation()
    # Convert to NumPy array if necessary
    first_frame = np.asarray(first_frame)
    if first_frame.dtype != np.uint8:
        first_frame = first_frame.astype(np.uint8)
    height, width = first_frame.shape[:2]
    x1,y1=trajectory[i]
    array=[first_frame.copy()]
    for i in range(1,len(trajectory)):
        x2,y2=trajectory[i]
        velx = x2 - x1
        vely = y2 - y1
        env.moveAgent(velx,vely) #move agent by velocity
        view_at_point=env.getObservation() #get the ant vision
        view_at_point = np.asarray(view_at_point)
        # Make sure OpenCV can write the frame
        if view_at_point.dtype != np.uint8:
            view_at_point = view_at_point.astype(np.uint8)

        # If the observation is grayscale, convert to BGR
        if len(view_at_point.shape) == 2:
            view_at_point = cv2.cvtColor(
                view_at_point,
                cv2.COLOR_GRAY2BGR
            )

        # Resize if necessary
        if (view_at_point.shape[1], view_at_point.shape[0]) != (width, height):
            view_at_point = cv2.resize(
                view_at_point,
                (width, height)
            )

        # Save frame to video
        array.append(view_at_point)
        #save to video stream
    #close video
    np.save(filename,np.array(array).astype(np.uint8))
