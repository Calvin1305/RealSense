import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import os
import datetime

def findDevices():
    ''' 
    Function that finds ands lists all connected realsense devices.
    INPUTS: None
    OUTPUTS:
        - serials: serial numbers of the cameras
        - ctx: camera controller "context" object needed to operate the cameras
    '''
    
    # creates librealsense context for managing devices
    ctx = rs.context() 
    serials = []
    # if more than one device was found
    if (len(ctx.devices) > 0):
        # loops over the devices and prints their infos
        for dev in ctx.devices:
            print ('Found device: ', \
                    dev.get_info(rs.camera_info.name), ' ', \
                    dev.get_info(rs.camera_info.serial_number))
            # appends to the "serials" list the cameras serial numbers
            serials.append(dev.get_info(rs.camera_info.serial_number))
    else:
        print("No Intel Device connected")
        
    return serials, ctx

def enableDevices(serials, ctx, resolution_width = 640, resolution_height = 480, frame_rate = 30):
    '''
    Function that enables all cameras by serial number using the specified options.
    Please note that possible values for camera's width and height are different
    for color and depth streams. The only two resolutions that match for the two 
    streams are 640x480 and 1280x720. For calibration purposes, it is best to
    keep the same resolution for each camera in the pipeline (calibration works
    anyway even if resolutions are different, but it is easier to keep them equal)
    
    INPUTS:
        - serials: list containing the cameras serial numbers as str
        - ctx: the cameras controller as pyrealsense2 context object
        - resolution_width: the camera resolution width 
        - resolution_width: the camera resolution height
        - frame_rate: the camera objective frame rate. 15 or 30 fps are the best choices
        
    OUTPUTS:
        - pipelines: list of pyrealsense2 streaming pipelines started by using 
                     the cameras' serial numbers and input specs
    '''
    
    pipelines = []
    # loops over the serial numbers in serials
    for serial in serials:
        # starts a pipeline
        pipe = rs.pipeline(ctx)
        # creates a configuration object
        cfg = rs.config()
        # enables the current device according to the serial number
        cfg.enable_device(serial)
        # enables the streaming pipeline for DEPTH using 16bit to represent the data
        cfg.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
        # enables the streaming pipeline for COLOR using RGB 8bit
        cfg.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.rgb8, frame_rate)
        
        # starts the pipeline for the current configuration
        pipe.start(cfg)
        # appends the pipeline coupled with the camera's serial
        pipelines.append([serial,pipe])
        
    return pipelines    

def visualize(pipelines, saving_dir, i):
    '''
    Function that shows the cameras' streams (left: color, right: depth with jet colormap)
    in real time (using the cameras' FPS settings).
    
    INPUTS:
        - pipelines: the list of streaming pipelines coupled with their serial number
        - saving_dir: full path to the saving directory in which the saved frames will be stored
        - i: image counter passed from the outside loop to save multiple frames
    
    OUTPUTS:
        - ext: exit flag (bool), True if user pressed Q or ESC
        - saved: saved frame flag (bool), True if user pressed S to save the current frame
    '''
    
    # WARNING: the color stream and the depth stream are NOT aligned by default
    # so the code first defines that the alignment process must align the depth map
    # to the color image using internal calibration parameters and functions stored 
    # in the cameras' object. "align" is therefore an object
    align_to = rs.stream.color
    align = rs.align(align_to)
    ext = False
    saved = []
    color_imgs = []
    depth_imgs = []
    pcs = []
    frms = []

    # loops over items in pipelines and separates them since they are saved as a list of lists
    for (device,pipe) in pipelines:
        # acquires color and depth frames
        frames = pipe.wait_for_frames()
        # align the depth frame to the color frame by calling the process method
        aligned_frames = align.process(frames)

        # extracts the aligned frames from the object
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # validate that both frames are valid (alignment may fail)
        if not aligned_depth_frame or not color_frame:
            continue
        
        # compute point cloud
        pc = rs.pointcloud()
        pc.map_to(color_frame)
        pointcloud = pc.calculate(aligned_depth_frame)
        
        
        # transform the images into arrays of any format (needed to keep the original data format)
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        # opencv native format is BGR (blue first, red last) so to save color images properly
        # we need to convert the color image to BGR here
        color_image = cv.cvtColor(color_image, cv.COLOR_RGB2BGR)

        # render the frames so they can be shown by OpenCV
        # the depth map has jet colormap applied on it to represent distance
        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)
        # stacks the two frames horizontally
        images = np.hstack((color_image, depth_colormap))
        
        # now shows the stacked frames
        cv.imshow('RealSense' + device, images)
        key = cv.waitKey(1)
        # press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv.destroyAllWindows()
            ext = True
            
        # save images and depth maps from both cameras by pressing 's'   
        if key==115 or len(saved) > 0:
            print('Entered for device: ' + str(device))
            depth_image = depth_image.copy()
            color_image = color_image.copy()
            #pointcloud = pointcloud.copy()
            color_imgs.append(color_image)
            depth_imgs.append(depth_image)
            frms.append(color_frame)
            pcs.append(pointcloud)
            saved.append(True)
    
    if len(saved) == 2 and len(color_imgs) == len(pipelines):
        for imcounter in range(0, len(pipelines)):
            device = pipelines[imcounter][0]
            cv.imwrite(os.path.join(saving_dir, str(device) + '_aligned_depth_' + str(i) +'.png'), depth_imgs[imcounter])
            cv.imwrite(os.path.join(saving_dir, str(device) + '_aligned_color_' + str(i) +'.png'), color_imgs[imcounter])
            current_pc = pcs[imcounter]
            
            current_pc.export_to_ply(os.path.join(saving_dir, str(device) + '_pointcloud_' + str(i) +'.ply'), frms[imcounter])
            #np.savetxt(os.path.join(saving_dir, str(device) + '_xyz_' + str(i) +'.txt'),vtx,delimiter=';')
            #np.savetxt(os.path.join(saving_dir, str(device) + '_tex_' + str(i) +'.txt'),colors,delimiter=';')
            print('[INFO] Device ' + str(device) + ' - Saved image ' + str(i))
    
    return ext, saved
            
def pipelineStop(pipelines):
    '''
    Function that stops the devices and releases the resource.
    
    INPUTS:
        - pipelines: the list containing the serial number and pipeline object
                     of each camera
    
    OUTPUTS: None
    '''
    
    for (device,pipe) in pipelines:
        # stop streaming for each item in the pipelines list
        pipe.stop() 
        

if __name__ == '__main__': 
    '''
    Main program. This gets executed when the program is launched but it is not
    when the program is imported as a library elsewhere.
    '''

    # inits the environment according to the connected cameras    
    serials, ctx = findDevices()
    
    # the only two resolutions that match for the two streams are 640x480 and 1280x720
    resolution_width = 640 # pixels
    resolution_height = 480 # pixels
    frame_rate = 30 # fps
    
    # creates a saving directory in the same folder the script is
    # named as the day and date of the acquisition YYYYMMDD_HHMMSS
    foldername = datetime.datetime.now()
    foldername = foldername.strftime('%Y%m%d_%H%M%S')
    # get current dir
    cwd = os.getcwd()
    saving_dir = os.path.join(cwd, foldername)
    # creates directory if it doesn't exist
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    
    # starts the pipelines with the user configs
    pipelines = enableDevices(serials, ctx, resolution_width, resolution_height, frame_rate)
    print('[INFO] Press ESC or Q to close the image window.')
    print('[INFO] Save images and depth maps from both cameras by pressing S. Only one frame will be saved for each pressing.')
    # starts the acquisition loop for all cameras at the same time
    i = 0
    if len(serials) > 0:    
        try:
            while True:
                ext, saved = visualize(pipelines, saving_dir, i)
                if len(saved) == 2:
                    i = i + 1
                if ext == True:
                    print('[INFO] Program closing...')
                    break
        finally:
            pipelineStop(pipelines)