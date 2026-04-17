import cv2 as cv
import glob
import numpy as np
import os

def calibrate_camera_for_intrinsic_parameters(images_names, calibration_settings):
    ''' Calibrate single camera to obtain camera intrinsic parameters from saved frames.
    
    INPUTS:
        - images_names: list containing the image paths
        - calibration_settings: dictionary containing the calibration parameters
        
    OUTPUTS:
        - K: intrinsic matrix of the camera resulting from the calibration
        - D: distorsion coefficients resulting from the calibration
    '''
    
    #read all frames
    images = [cv.imread(imname, 1) for imname in images_names]

    # criteria used by checkerboard pattern detector
    # change this if the code can't find the checkerboard
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    
    # WARNING: the software actually needs to know the intersections between 
    # squares, not the number of squares. This corresponds to the number of squares
    # minus 1, so to improve simplicity we do this here
    rows = calibration_settings['checkerboard_rows']-1
    columns = calibration_settings['checkerboard_columns']-1
    world_scaling = calibration_settings['checkerboard_box_size']
    conv_size = calibration_settings['conv_size']

    # gets the coordinates of squares in the checkerboard world space (meters)
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling*objp

    # get frame dimensions
    # the two frames from cam0 and cam1 should be the same!
    width = images[0].shape[1]
    height = images[0].shape[0]

    # get the pixel and world coordinates of checkerboards
    imgpoints = [] # 2d points in image plane (px)
    objpoints = [] # 3d points in real world space (m)

    for i, frame in enumerate(images):
        print('-- Analyzing image ' + str(i))
        # the sw works in grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)
        
        # if a checkerboard was found
        if ret == True:
            # calculate the intersection points           

            # opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            # draws the corners just found in the image
            cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
            # draws the helper text
            cv.putText(frame, 'If detected points are poor, press "s" to skip this sample', (25, 25), cv.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255), 2)

            # shows the image to the user, if ok press whatever, if to be skipped press "S"
            cv.imshow('img', frame)
            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print('++ Skipping frame ' + str(i))
                continue
            
            # only appends data if the previous if skipping didn't happen
            objpoints.append(objp)
            imgpoints.append(corners)

    # close all windows
    cv.destroyAllWindows()
    # now with all the points just saved attempts to calibrate the camera
    ret, K, D, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('=== CALIBRATION RESULTS ===')
    print('RMSE:', ret)
    print('Camera matrix K:\n', K)
    print('Distortion coeffs:', D)
    print('===========================')

    return K, D

def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name, savedir):
    ''' Function to save the camera intrinsic parameters to file.
    
    INPUTS:
        - camera_matrix: intrinsic camera K
        - distortion_coefs: distorsion coefficients D
        - camera_name: serial name of the camera
        - savedir: directory in which the new subfolder containing these data will be saved
        
    OUTPUTS:
        - None
    '''

    # create folder if it does not exist
    if not os.path.exists(os.path.join(savedir,'camera_parameters')):
        os.mkdir(os.path.join(savedir,'camera_parameters'))
        
    # creates the filename
    out_filename = os.path.join(savedir, 'camera_parameters', camera_name + '_intrinsics.txt')
    outf = open(out_filename, 'w')

    outf.write('Camera matrix K:\n')
    # saves each element of the camera matrix object
    for l in camera_matrix:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    # saves each element of the distorsion matrix object
    outf.write('Distorsion coeffs D:\n')
    for en in distortion_coefs[0]:
        outf.write(str(en) + ' ')
    outf.write('\n')

def stereo_calibrate(K0, D0, K1, D1, c0_images_names, c1_images_names, calibration_settings):
    ''' Function that applies stereo calibration on the two pairs of data.
    Points in the camera0 space will be moved to the camera1 space by applying the resulting R and T.
    
    INPUTS:
        - K0, K1: intrinsic camera matrix K for camera master (0) and slave (1)
        - D0, D1: distorsion coefficients D for camera master (0) and slave (1)
        - c0_images_names, c1_images_names: filenames of the color images for both cameras
        - calibration_settings: dictionary containing the calibration pattern parameters
    
    OUTPUTS:
        - R: rotation matrix resulting from the calibration procedure
        - T: translation vector resulting from the calibration procedure
    '''

    # open color images of both cameras, they are paired by name since we 
    # used the same numbering in the acquisition software
    c0_images = [cv.imread(imname, 1) for imname in c0_images_names]
    c1_images = [cv.imread(imname, 1) for imname in c1_images_names]
    
    names = [c0_images_names[0], c1_images_names[0]]
    
    # finds the serial numbers
    serials = []
    for item in names:
        s = os.path.basename(item)
        s = s.split('_')
        serials.append(s[0])
    
    print('[INFO] Stereo calibration starting. Resulting R and T will move points from camera ' + serials[0] + ' into camera ' + serials[1] + ' space')

    # change these params if the stereo calibration result is not good enough
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    # calibration pattern settings, same as the intrinsic calibration
    rows = calibration_settings['checkerboard_rows']-1
    columns = calibration_settings['checkerboard_columns']-1
    world_scaling = calibration_settings['checkerboard_box_size']
    conv_size = calibration_settings['conv_size']

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    # again, the frames of both cameras should be the same size
    width = c0_images[0].shape[1]
    height = c0_images[0].shape[0]

    # pixel coordinates of checkerboards for both cameras
    imgpoints_left = [] # 2d points in image plane for LEFT camera
    imgpoints_right = [] # 2d points in image plane for RIGHT camera

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space

    # iterates in a loop over couples of images
    for frame0, frame1 in zip(c0_images, c1_images):
        # converts them to grayscale
        gray1 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        # finds the chessboard
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)
        
        # the chessboard must be found in both images, otherwise it skips the couple
        if c_ret1 == True and c_ret2 == True:
            
            # finds the intersections for both checkerboards
            corners1 = cv.cornerSubPix(gray1, corners1, conv_size, (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, conv_size, (-1, -1), criteria)
            
            # draws the points for each image
            cv.drawChessboardCorners(frame0, (rows,columns), corners1, c_ret1)
            cv.imshow('img_slave_'+serials[0], frame0)

            cv.drawChessboardCorners(frame1, (rows,columns), corners2, c_ret2)
            cv.imshow('img_master_'+serials[1], frame1)
            k = cv.waitKey(0)
            
            # to accept the couple press whatever, if you wanna skip press 'S'
            if k & 0xFF == ord('s'):
                print('++ Skipping coupled frames')
                continue
            
            # only appends results if couple was not skipped
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    cv.destroyAllWindows()
    # this flag means that we are gonna pass the intrinsic matrix to the 
    # algorithm for both camera instead of calculating it
    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    # applies stereo calibration
    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv.stereoCalibrate(objpoints, 
                                                                 imgpoints_left, 
                                                                 imgpoints_right, 
                                                                 K0, D0,
                                                                 K1, D1, 
                                                                 (width, height), 
                                                                 criteria = criteria, 
                                                                 flags = stereocalibration_flags)

    print('=== STEREO CAMERA RESULTS ===')
    print('=== camera ' + serials[0] + ' moved to camera ' + serials[1] + ' ===')
    print('RMSE: ', ret)
    print('Rotation matrix: ' + str(R))
    print('Translation matrix: ' + str(T))
    print('=============================') 
    print('Complete RT in homogeneous coordinates')
    print(str(R[0,0]) + '\t' + str(R[0,1]) + '\t' + str(R[0,2]) + '\t' + str(T[0][0]))
    print(str(R[1,0]) + '\t' + str(R[1,1]) + '\t' + str(R[1,2]) + '\t' + str(T[0][1]))
    print(str(R[2,0]) + '\t' + str(R[2,1]) + '\t' + str(R[2,2]) + '\t' + str(T[0][2]))
    print(str(0.0) + '\t' + str(0.0) + '\t' + str(0.0) + '\t' + str(1.0))
    print('=============================')    
    return R, T

def save_extrinsic_calibration_parameters(R, T, savedir):
    ''' Function to save the camera intrinsic parameters to file.
    
    INPUTS:
        - R: rotation matrix resulting from the extrinsic calibration
        - D: translation vector resulting from the extrinsic calibration
        - savedir: directory in which the new subfolder containing these data will be saved
        
    OUTPUTS:
        - None
    '''
    
    # create folder if it does not exist
    if not os.path.exists(os.path.join(savedir,'camera_parameters')):
        os.mkdir(os.path.join(savedir,'camera_parameters'))

    extrinsics_filename = os.path.join(savedir, 'camera_parameters', 'slave-to-master_RT.txt')
    outf = open(extrinsics_filename, 'w')

    outf.write('R:\n')
    for l in R:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

if __name__ == '__main__':
    
    # define the test folder name
    testfolder = 'test'
    # define the calibration pattern specs
    # rows and cols: here put the exact number of squares in the checkerboard
    # rows are always the shorter size
    # box_size is the size of the single square in meters
    # convolution kernel "conv_size" is needed to find the checkerboard
    # usual values are 9, 11, 15: use 9 or lower when using small checkerboards
    # use 11 or higher if using big checkerboards (a few centimeters per square)
    calibration_settings = {'checkerboard_rows': 7, 
                            'checkerboard_columns': 10, 
                            'checkerboard_box_size': 0.0115,
                            'conv_size': (6,6)}
    
    # getting the full path to the test folder
    # using the "current working directory" path which is the path 
    # from which this code is launched from
    print('[WARNING] Keep in mind that the current directory MUST contain the test folder!')
    print('[INFO] Current working directory is: ' + str(os.getcwd()))
    PATH = os.path.join(os.getcwd(), testfolder)
    
    # first loads a pair to strip the serial numbers
    filenames = glob.glob(os.path.join(PATH,'*_color_0*'))
    serials = []
    for item in filenames:
        s = os.path.basename(item)
        s = s.split('_')
        serials.append(s[0])
        print('[INFO] Found data from device ID: ' + s[0])
        
    # now loads filenames separately from each device
    filenames = {}
    for s in serials:
        filenames[s] = glob.glob(os.path.join(PATH,s+'*_color_*'))        

    # calibro intrinsecamente le due camere, ovvero stimo la matrice K per ognuna
    intrinsics = {}
    for s in serials:
        print('[INFO] Computing intrinsics for camera ' + s)
        print('[INFO] Press s to skip the frame if the quality is poor')
        K, D = calibrate_camera_for_intrinsic_parameters(filenames[s], calibration_settings)
        # saves intrinsic params
        save_camera_intrinsics(K, D, s, PATH)
        intrinsics[s] = [K, D]

    print('[INFO] Check the position of the red "O". If wrong, press s to skip the pair.')
    R, T = stereo_calibrate(intrinsics[serials[0]][0], intrinsics[serials[0]][1], 
                            intrinsics[serials[1]][0], intrinsics[serials[1]][1], 
                            filenames[serials[0]], filenames[serials[1]], 
                            calibration_settings)
    # saves result
    save_extrinsic_calibration_parameters(R, T, PATH)
   
