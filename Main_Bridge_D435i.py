
import pyrealsense2 as rs
import mediapipe as mp
import numpy as np
import cv2
import datetime as dt
import csv
import pandas as pd

def convert(array):
  result = [array[0], array[1], array[2]]
  return result

def Detect():
    
    ##====== Camera ======##
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (20, 100)
    fontScale = .5
    color = (0,150,255)
    thickness = 1

    # ====== Realsense ======
    realsense_ctx = rs.context()
    connected_devices = [] # List of serial numbers for present cameras
    for i in range(len(realsense_ctx.devices)):
        detected_camera =  realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
        print(f"{detected_camera}")
        connected_devices.append(detected_camera)
        device = connected_devices[0] # In this example we are only using one camera
        pipeline = rs.pipeline()
        config = rs.config()
        background_removed_color = 153 # Grey


    # ====== Mediapipe ======
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose


    # ====== Enable Streams ======
    config.enable_device(device)
    # # For worse FPS, but better resolution:
    stream_res_x = 1280
    stream_res_y = 720
    # # For better FPS. but worse resolution:
    #stream_res_x = 640
    #stream_res_y = 480

    stream_fps = 30
    config.enable_stream(rs.stream.depth, stream_res_x, stream_res_y, rs.format.z16, stream_fps)
    config.enable_stream(rs.stream.color, stream_res_x, stream_res_y, rs.format.bgr8, stream_fps)
    config.enable_stream(rs.stream.accel)
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)


    # ====== Get depth Scale ======
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print(f"\tDepth Scale for Camera SN {device} is: {depth_scale}")# ====== Set clipping distance ======
    clipping_distance_in_meters = 2
    clipping_distance = clipping_distance_in_meters / depth_scale
    print(f"\tConfiguration Successful for SN {device}")



    # For static images:
    IMAGE_FILES = []
    BG_COLOR = (192, 192, 192) # gray
    with mp_pose.Pose(
        static_image_mode = False,
        model_complexity = 2,
        smooth_landmarks = True,
        enable_segmentation = True,
        smooth_segmentation = True,  
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8) as pose:
        for idx, file in enumerate(IMAGE_FILES):
            image = cv2.imread(file)
            image_height, image_width, _ = image.shape
            # Convert the BGR image to RGB before processing.
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            if not results.pose_landmarks:
                continue
            print(
                f'Nose coordinates: ('
                f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
                f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
            )

            annotated_image = image.copy()
            # Draw segmentation on the image.
            # To improve segmentation around boundaries, consider applying a joint
            # bilateral filter to "results.segmentation_mask" with "image".
            condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
            bg_image = np.zeros(image.shape, dtype=np.uint8)
            bg_image[:] = BG_COLOR
            annotated_image = np.where(condition, annotated_image, bg_image)
            # Draw pose landmarks on the image.
            mp_drawing.draw_landmarks(
                annotated_image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            cv2.imwrite('/tmp/annotated_image' + str(idx) + '.png', annotated_image)
            # Plot pose world landmarks.
            mp_drawing.plot_landmarks(
                results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

        features = ["Origin_X", "Origin_Y", "Origin_Z", "Left_Shoulder_X", "Left_Shoulder_Y", "Left_Shoulder_Z", "Right_Shoulder_X", "Right_Shoulder_Y", "Right_Shoulder_Z", "Left_Hip_X", "Left_Hip_Y", "Left_Hip_Z", "Right_Hip_X", "Right_Hip_Y", "Right_Hip_Z", "Left_Knee_X", "Left_Knee_Y", "Left_Knee_Z", "Right_Knee_X", "Right_Knee_Y", "Right_Knee_Z", "Left_Ankle_X", "Left_Ankle_Y", "Left_Ankle_Z", "Right_Ankle_X", "Right_Ankle_Y", "Right_Ankle_Z", "Left Knee Angle", "Right Knee Angle", "Left Yaw Angle", "Right Yaw Angle", "Left Pitch Angle", "Right Pitch Angle", "Left Roll Angle", "Right Roll Angle"]
        liveData = pd.DataFrame(columns = features)
        # liveData.set_index("Time")

    # For webcam input:
    
        while True:
            start_time = dt.datetime.today().timestamp() # Necessary for FPS calculations

            # Get and align frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            Accel_frame = frames[2].as_motion_frame().get_motion_data()

            accel = [Accel_frame.x, Accel_frame.y, Accel_frame.z]
            Cam_Ang_X = np.rad2deg(np.arccos(np.dot(np.array([0,1]),np.array([accel[1],accel[2]]))/(np.linalg.norm(np.array([accel[1],accel[2]])))))-90
            #print(Cam_Ang_X)
            

            if not aligned_depth_frame or not color_frame:
                continue

            # Process images
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            depth_image_flipped = cv2.flip(depth_image,1)
            color_image = np.asanyarray(color_frame.get_data())

            depth_image_3d = np.dstack((depth_image,depth_image,depth_image))
            #background_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), background_removed_color, color_image)

            #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            #image = cv2.flip(background_removed,1)
            #image = cv2.flip(color_image,1)
            image = color_image

            #color_image = cv2.flip(color_image,1)
            color_images_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)


            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = pose.process(color_images_rgb)

            if results.pose_world_landmarks:
                org2 = (20, org[1]+(20*(i+1)))
                # Draw the pose annotation on the image.
                image.flags.writeable = True
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                


                ##Collecting Position

                #Left Shoulder
                L_Shoulder_position = results.pose_landmarks.landmark[11]
                L_Shoulder_x = L_Shoulder_position.x*len(depth_image_flipped[11])
                L_Shoulder_y = L_Shoulder_position.y*len(depth_image_flipped)
                if L_Shoulder_x >= len(depth_image_flipped[11]):
                    L_Shoulder_x = len(depth_image_flipped[11]) - 1
                if L_Shoulder_y >= len(depth_image_flipped):
                    L_Shoulder_y = len(depth_image_flipped) - 1
                
                L_Shoulder_z = depth_image_flipped[int(L_Shoulder_y),int(L_Shoulder_x)] * depth_scale *1000 # meters
                L_Shoulder_y *= 2*L_Shoulder_z*np.tan(np.deg2rad(42/2))/stream_res_y
                L_Shoulder_x *= 2*L_Shoulder_z*np.tan(np.deg2rad(69/2))/stream_res_x

                L_Shoulder_raw = np.array([L_Shoulder_x, L_Shoulder_y, L_Shoulder_z])
                L_Shoulder_visibility = L_Shoulder_position.visibility
                
                

                #Right Shoulder
                R_Shoulder_position = results.pose_landmarks.landmark[12]
                R_Shoulder_x = R_Shoulder_position.x*len(depth_image_flipped[12])
                R_Shoulder_y = R_Shoulder_position.y*len(depth_image_flipped)
                if R_Shoulder_x >= len(depth_image_flipped[12]):
                    R_Shoulder_x = len(depth_image_flipped[12]) - 1
                if R_Shoulder_y >= len(depth_image_flipped):
                    R_Shoulder_y = len(depth_image_flipped) - 1
                
                R_Shoulder_z = depth_image_flipped[int(R_Shoulder_y),int(R_Shoulder_x)] * depth_scale *1000 # meters
                R_Shoulder_y *= 2*R_Shoulder_z*np.tan(np.deg2rad(42/2))/stream_res_y
                R_Shoulder_x *= 2*R_Shoulder_z*np.tan(np.deg2rad(69/2))/stream_res_x
                
                R_Shoulder_raw = np.array([R_Shoulder_x, R_Shoulder_y, R_Shoulder_z])
                R_Shoulder_visibility = R_Shoulder_position.visibility
                
                
        

                #Left_Hip
                L_Hip_position = results.pose_landmarks.landmark[23]
                L_Hip_x = L_Hip_position.x*len(depth_image_flipped[23])
                L_Hip_y = L_Hip_position.y*len(depth_image_flipped)
                if L_Hip_x >= len(depth_image_flipped[23]):
                    L_Hip_x = len(depth_image_flipped[23]) - 1
                if L_Hip_y >= len(depth_image_flipped):
                    L_Hip_y = len(depth_image_flipped) - 1
                
                L_Hip_z = depth_image_flipped[int(L_Hip_y),int(L_Hip_x)] * depth_scale *1000 # meters
                L_Hip_y *= 2*L_Hip_z*np.tan(np.deg2rad(42/2))/stream_res_y
                L_Hip_x *= 2*L_Hip_z*np.tan(np.deg2rad(69/2))/stream_res_x
                
                L_Hip_raw = np.array([L_Hip_x, L_Hip_y, L_Hip_z])
                L_Hip_visibility = L_Hip_position.visibility
                
                



                #Right_Hip
                R_Hip_position = results.pose_landmarks.landmark[24]
                R_Hip_x = R_Hip_position.x*len(depth_image_flipped[24])
                R_Hip_y = R_Hip_position.y*len(depth_image_flipped)
                if R_Hip_x >= len(depth_image_flipped[24]):
                    R_Hip_x = len(depth_image_flipped[24]) - 1
                if R_Hip_y >= len(depth_image_flipped):
                    R_Hip_y = len(depth_image_flipped) - 1
                
                R_Hip_z = depth_image_flipped[int(R_Hip_y),int(R_Hip_x)] * depth_scale *1000 # meters
                R_Hip_y *= 2*R_Hip_z*np.tan(np.deg2rad(42/2))/stream_res_y
                R_Hip_x *= 2*R_Hip_z*np.tan(np.deg2rad(69/2))/stream_res_x
                
                R_Hip_raw = np.array([R_Hip_x, R_Hip_y, R_Hip_z])
                R_Hip_visibility = R_Hip_position.visibility
                
                


                #Left_Knee
                L_Knee_position = results.pose_landmarks.landmark[25]
                L_Knee_x = L_Knee_position.x*len(depth_image_flipped[25])
                L_Knee_y = L_Knee_position.y*len(depth_image_flipped)
                if L_Knee_x >= len(depth_image_flipped[25]):
                    L_Knee_x = len(depth_image_flipped[25]) - 1
                if L_Knee_y >= len(depth_image_flipped):
                    L_Knee_y = len(depth_image_flipped) - 1               
                
                L_Knee_z = depth_image_flipped[int(L_Knee_y),int(L_Knee_x)] * depth_scale *1000 # meters
                L_Knee_y *= 2*L_Knee_z*np.tan(np.deg2rad(42/2))/stream_res_y
                L_Knee_x *= 2*L_Knee_z*np.tan(np.deg2rad(69/2))/stream_res_x
                
                L_Knee_raw = np.array([L_Knee_x, L_Knee_y, L_Knee_z])
                L_Knee_visibility = L_Knee_position.visibility
                
                



                #Right_Knee
                R_Knee_position = results.pose_landmarks.landmark[26]
                R_Knee_x = R_Knee_position.x*len(depth_image_flipped[26])
                R_Knee_y = R_Knee_position.y*len(depth_image_flipped)
                if R_Knee_x >= len(depth_image_flipped[26]):
                    R_Knee_x = len(depth_image_flipped[26]) - 1
                if R_Knee_y >= len(depth_image_flipped):
                    R_Knee_y = len(depth_image_flipped) - 1
                
                R_Knee_z = depth_image_flipped[int(R_Knee_y),int(R_Knee_x)] * depth_scale *1000 # meters
                R_Knee_y *= 2*R_Knee_z*np.tan(np.deg2rad(42/2))/stream_res_y
                R_Knee_x *= 2*R_Knee_z*np.tan(np.deg2rad(69/2))/stream_res_x
                
                R_Knee_raw = np.array([R_Knee_x, R_Knee_y, R_Knee_z])
                R_Knee_visibility = R_Knee_position.visibility
                
                
                
               

                #Left_Ankle
                L_Ankle_position = results.pose_landmarks.landmark[27]
                L_Ankle_x = L_Ankle_position.x*len(depth_image_flipped[27])
                L_Ankle_y = L_Ankle_position.y*len(depth_image_flipped)
                if L_Ankle_x >= len(depth_image_flipped[23]):
                    L_Ankle_x = len(depth_image_flipped[23]) - 1
                if L_Ankle_y >= len(depth_image_flipped):
                    L_Ankle_y = len(depth_image_flipped) - 1
                
                L_Ankle_z = depth_image_flipped[int(L_Ankle_y),int(L_Ankle_x)] * depth_scale *1000 # meters
                L_Ankle_y *= 2*L_Ankle_z*np.tan(np.deg2rad(42/2))/stream_res_y
                L_Ankle_x *= 2*L_Ankle_z*np.tan(np.deg2rad(69/2))/stream_res_x
                
                L_Ankle_raw = np.array([L_Ankle_x, L_Ankle_y, L_Ankle_z])
                L_Ankle_visibility = L_Ankle_position.visibility
                
                



                #Right_Ankle
                R_Ankle_position = results.pose_landmarks.landmark[28]
                R_Ankle_x = R_Ankle_position.x*len(depth_image_flipped[28])
                R_Ankle_y = R_Ankle_position.y*len(depth_image_flipped)
                if R_Ankle_x >= len(depth_image_flipped[24]):
                    R_Ankle_x = len(depth_image_flipped[24]) - 1
                if R_Ankle_y >= len(depth_image_flipped):
                    R_Ankle_y = len(depth_image_flipped) - 1

                R_Ankle_z = depth_image_flipped[int(R_Ankle_y),int(R_Ankle_x)] * depth_scale *1000 # meters
                R_Ankle_y *= 2*R_Ankle_z*np.tan(np.deg2rad(42/2))/stream_res_y
                R_Ankle_x *= 2*R_Ankle_z*np.tan(np.deg2rad(69/2))/stream_res_x
                
                R_Ankle_raw = np.array([R_Ankle_x, R_Ankle_y, R_Ankle_z])
                R_Ankle_visibility = R_Ankle_position.visibility
                
                

                #Print Raw Data
                #print("L_Shoulder_Pos", L_Shoulder_raw)
                #print("R_Shoulder_Pos", R_Shoulder_raw)
                #print("L_Hip_Pos", L_Hip_raw)
                #print("R_Hip_Pos", R_Hip_raw)
                #print("L_Knee_Pos", L_Knee_raw)
                #print("R_Knee_Pos", R_Knee_raw)
                #print("L_Ankle_Pos", L_Ankle_raw)
                #print("R_Ankle_Pos", R_Ankle_raw)


                #Print Visibility
                # print("L_Shoulder_visibility: ", L_Shoulder_visibility)
                # print("R_Shoulder_visibility: ", R_Shoulder_visibility)
                # print("L_Hip_visibility: ", L_Hip_visibility)
                # print("R_Hip_visibility: ", R_Hip_visibility)
                # print("L_Knee_visibility: ", L_Knee_visibility)
                # print("R_Knee_visibility: ", R_Knee_visibility)
                # print("L_Ankle_visibility: ", L_Ankle_visibility)
                # print("R_Ankle_visibility: ", R_Ankle_visibility)

                
                


                threshold = 0.75
                qualiy_check = 0
                if L_Shoulder_visibility > threshold and R_Shoulder_visibility > threshold and L_Hip_visibility > threshold and R_Hip_visibility > threshold and L_Knee_visibility > threshold and R_Knee_visibility > threshold and L_Ankle_visibility > threshold and R_Ankle_visibility > threshold:

                    quality_check = 1

                    ###-----Rotate_Methods-----###
                    R_Matrix = np.linalg.inv(np.array([
                                            [-1,                              0,                              0],
                                            [ 0,  np.sin(np.deg2rad(Cam_Ang_X)), -np.cos(np.deg2rad(Cam_Ang_X))],
                                            [ 0, -np.cos(np.deg2rad(Cam_Ang_X)), -np.sin(np.deg2rad(Cam_Ang_X))]]))
                    
                    L_Shoulder_raw = np.dot(R_Matrix, L_Shoulder_raw)
                    R_Shoulder_raw = np.dot(R_Matrix, R_Shoulder_raw)
                    L_Hip_raw = np.dot(R_Matrix, L_Hip_raw)
                    R_Hip_raw = np.dot(R_Matrix, R_Hip_raw)
                    L_Knee_raw = np.dot(R_Matrix, L_Knee_raw)
                    R_Knee_raw = np.dot(R_Matrix, R_Knee_raw)
                    L_Ankle_raw = np.dot(R_Matrix, L_Ankle_raw)
                    R_Ankle_raw = np.dot(R_Matrix, R_Ankle_raw)

                    origin_Raw = (L_Shoulder_raw+R_Shoulder_raw)/2

                    L_Shoulder = L_Shoulder_raw - origin_Raw
                    R_Shoulder = R_Shoulder_raw - origin_Raw
                    L_Hip = L_Hip_raw - origin_Raw
                    R_Hip = R_Hip_raw - origin_Raw
                    L_Knee = L_Knee_raw - origin_Raw
                    R_Knee = R_Knee_raw - origin_Raw
                    L_Ankle = L_Ankle_raw - origin_Raw
                    R_Ankle = R_Ankle_raw - origin_Raw



                    ###-----T_Matrix_Methods-----###
                    #origin_Raw = (L_Shoulder_raw+R_Shoulder_raw)/2
                    # T_Matrix = np.linalg.inv(np.array([[1,                              0,                              0, origin_Raw[0]],
                    #                      [             0,   np.sin(np.deg2rad(Cam_Ang_X)), -np.cos(np.deg2rad(Cam_Ang_X)), origin_Raw[1]],
                    #                      [             0,  -np.cos(np.deg2rad(Cam_Ang_X)), -np.sin(np.deg2rad(Cam_Ang_X)), origin_Raw[2]],
                    #                      [             0,                              0,                              0,             1]]))
                    
                    # #print(T_Matrix)
                    # L_Shoulder = np.delete(np.dot(T_Matrix, np.append(L_Shoulder_raw,1)), -1)
                    # R_Shoulder = np.delete(np.dot(T_Matrix, np.append(R_Shoulder_raw,1)), -1)
                    # L_Hip = np.delete(np.dot(T_Matrix, np.append(L_Hip_raw,1)), -1)
                    # R_Hip = np.delete(np.dot(T_Matrix, np.append(R_Hip_raw,1)), -1)
                    # L_Knee = np.delete(np.dot(T_Matrix, np.append(L_Knee_raw,1)), -1)
                    # R_Knee = np.delete(np.dot(T_Matrix, np.append(R_Knee_raw,1)), -1)
                    # L_Ankle = np.delete(np.dot(T_Matrix, np.append(L_Ankle_raw,1)), -1)
                    # R_Ankle = np.delete(np.dot(T_Matrix, np.append(R_Ankle_raw,1)), -1)


                    ##--Angle Calculation--##
                    L_Shoulder2R_Ankle = R_Ankle-L_Shoulder
                    R_Shoulder2L_Ankle = L_Ankle-R_Shoulder
                    Body_vec = np.cross(R_Shoulder2L_Ankle,L_Shoulder2R_Ankle)

                    ##Left
                    L_Thigh = L_Knee-L_Hip
                    L_Shank = L_Knee-L_Ankle
                    L_proj_Thigh = np.append(np.delete(L_Thigh,-1), L_Hip[2])
                    L2R_Hip = R_Hip-L_Hip
                    
                    L_Knee_vec = np.cross(L_Thigh,L_Shank)
                    
                    L_Knee_Angle = np.rad2deg(np.arccos(np.dot(L_Thigh,L_Shank)/(np.linalg.norm(L_Thigh)*np.linalg.norm(L_Shank))))
                    L_yaw = np.rad2deg(np.arccos(np.dot(L_proj_Thigh,L2R_Hip)/(np.linalg.norm(L_proj_Thigh)*np.linalg.norm(L2R_Hip))))
                    L_pitch = np.rad2deg(np.arccos(np.dot(L_proj_Thigh,L_Thigh)/(np.linalg.norm(L_proj_Thigh)*np.linalg.norm(L_Thigh))))
                    L_roll = np.rad2deg(np.arccos(np.dot(Body_vec,L_Knee_vec)/(np.linalg.norm(Body_vec)*np.linalg.norm(L_Knee_vec))))



                    ##Right
                    R_Thigh = R_Knee-R_Hip
                    R_Shank = R_Knee-R_Ankle
                    R_proj_Thigh = np.append(np.delete(R_Thigh,-1), R_Hip[2])
                    R2L_Hip = L_Hip-R_Hip

                    R_Knee_vec = np.cross(R_Thigh,R_Shank)



                    R_Knee_Angle = np.rad2deg(np.arccos(np.dot(R_Thigh,R_Shank)/(np.linalg.norm(R_Thigh)*np.linalg.norm(R_Shank))))
                    R_yaw = np.rad2deg(np.arccos(np.dot(R_proj_Thigh,R2L_Hip)/(np.linalg.norm(R_proj_Thigh)*np.linalg.norm(R2L_Hip))))
                    R_pitch = np.rad2deg(np.arccos(np.dot(R_proj_Thigh,R_Thigh)/(np.linalg.norm(R_proj_Thigh)*np.linalg.norm(R_Thigh))))
                    R_roll = np.rad2deg(np.arccos(np.dot(Body_vec,R_Knee_vec)/(np.linalg.norm(Body_vec)*np.linalg.norm(R_Knee_vec))))
                    


                    ## Data Plot
                    #Left
                    print("Left Shoulder", L_Shoulder)                   
                    print("Left Hip", L_Hip)                   
                    print("Left Knee", L_Knee)                   
                    print("Left Ankle", L_Ankle)
                    print("Left Knee Angle:", L_Knee_Angle)
                    print("Left Yaw Angle", L_yaw)
                    print("Left Pitch Angle", L_pitch)
                    print("Left Roll Angle", L_roll)
                    

                    #Right
                    print("Right Shoulder", R_Shoulder)
                    print("Right Hip", R_Hip)
                    print("Right Knee", R_Knee)
                    print("Right Ankle", R_Ankle)
                    print("Right Knee Angle:", R_Knee_Angle)
                    print("Right Yaw Angle", R_yaw)  
                    print("Right Pitch Angle", R_pitch)
                    print("Right Roll Angle", R_roll)  

                    #
                    OR = convert(origin_Raw)
                    LS = convert(L_Shoulder)
                    RS = convert(R_Shoulder)
                    LH = convert(L_Hip)
                    RH = convert(R_Hip)
                    LK = convert(L_Knee)
                    RK = convert(R_Knee)
                    LA = convert(L_Ankle)
                    RA = convert(R_Ankle)
                    liveData.loc[liveData.shape[0]] = [OR[0], OR[1], OR[2], LS[0], LS[1], LS[2], RS[0], RS[1], RS[2], LH[0], LH[1], LH[2], RH[0], RH[1], RH[2], LK[0], LK[1], LK[2], RK[0], RK[1], RK[2], LA[0], LA[1], LA[2], RA[0], RA[1], RA[2], L_Knee_Angle, R_Knee_Angle, L_yaw, R_yaw, L_pitch, R_pitch, L_roll, R_roll]
                    liveData.tail(150).to_csv('liveplot.csv')

                #image = cv2.flip(image,1)
                image = cv2.putText(image, f"Body Detected", org, font, fontScale, color, thickness, cv2.LINE_AA)
 
            else:
                #image = cv2.flip(image,1)
                image = cv2.putText(image,"No Body", org, font, fontScale, color, thickness, cv2.LINE_AA)
        
                
            
            # Display FPS
            time_diff = dt.datetime.today().timestamp() - start_time
            fps = int(1 / time_diff)
            org3 = (20, org[1] + 60)
            image = cv2.putText(image, f"FPS: {fps}", org3, font, fontScale, color, thickness, cv2.LINE_AA)
            name_of_window = 'SN: ' + str(device)
                

            # Display images
            
            cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
            cv2.imshow(name_of_window, image)

            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:

                print(f"User pressed break key for SN: {device}")
                print(f"Exiting loop for SN: {device}")
                print(f"Application Closing.")
                pipeline.stop()
                print(f"Application Closed.")
                break
            elif key & 0xFF == ord('a'):
                if quality_check == 1:
                    if pre_a == False:
                        pre_a = True
                        Start_UpA = dt.datetime.today().timestamp()
                        f = open('/home/joey/Documents/python/BridgeStand/Record/240266/Up02.csv', 'w')
                        writerA = csv.writer(f)
                        writerA.writerow(["Time", "Origin_X", "Origin_Y", "Origin_Z", "Left_Shoulder_X", "Left_Shoulder_Y", "Left_Shoulder_Z", "Right_Shoulder_X", "Right_Shoulder_Y", "Right_Shoulder_Z", "Left_Hip_X", "Left_Hip_Y", "Left_Hip_Z", "Right_Hip_X", "Right_Hip_Y", "Right_Hip_Z", "Left_Knee_X", "Left_Knee_Y", "Left_Knee_Z", "Right_Knee_X", "Right_Knee_Y", "Right_Knee_Z", "Left_Ankle_X", "Left_Ankle_Y", "Left_Ankle_Z", "Right_Ankle_X", "Right_Ankle_Y", "Right_Ankle_Z", "Left Knee Angle", "Right Knee Angle", "Left Yaw Angle", "Right Yaw Angle", "Left Pitch Angle", "Right Pitch Angle", "Left Roll Angle", "Right Roll Angle"])
                                    
                    A_Time_diff = dt.datetime.today().timestamp()-Start_UpA
                    OR = convert(origin_Raw)
                    LS = convert(L_Shoulder)
                    RS = convert(R_Shoulder)
                    LH = convert(L_Hip)
                    RH = convert(R_Hip)
                    LK = convert(L_Knee)
                    RK = convert(R_Knee)
                    LA = convert(L_Ankle)
                    RA = convert(R_Ankle)
                    writerA.writerow([A_Time_diff, OR[0], OR[1], OR[2], LS[0], LS[1], LS[2], RS[0], RS[1], RS[2], LH[0], LH[1], LH[2], RH[0], RH[1], RH[2], LK[0], LK[1], LK[2], RK[0], RK[1], RK[2], LA[0], LA[1], LA[2], RA[0], RA[1], RA[2], L_Knee_Angle, R_Knee_Angle, L_yaw, R_yaw, L_pitch, R_pitch, L_roll, R_roll])
            
            elif key & 0xFF == ord('s'):
                if quality_check == 1:
                    if pre_s == False:
                        pre_s = True
                        Start_UpS = dt.datetime.today().timestamp()
                        f = open('/home/joey/Documents/python/BridgeStand/Record/240266/Hold02.csv', 'w')
                        writerS = csv.writer(f)
                        writerS.writerow(["Time", "Origin_X", "Origin_Y", "Origin_Z", "Left_Shoulder_X", "Left_Shoulder_Y", "Left_Shoulder_Z", "Right_Shoulder_X", "Right_Shoulder_Y", "Right_Shoulder_Z", "Left_Hip_X", "Left_Hip_Y", "Left_Hip_Z", "Right_Hip_X", "Right_Hip_Y", "Right_Hip_Z", "Left_Knee_X", "Left_Knee_Y", "Left_Knee_Z", "Right_Knee_X", "Right_Knee_Y", "Right_Knee_Z", "Left_Ankle_X", "Left_Ankle_Y", "Left_Ankle_Z", "Right_Ankle_X", "Right_Ankle_Y", "Right_Ankle_Z", "Left Knee Angle", "Right Knee Angle", "Left Yaw Angle", "Right Yaw Angle", "Left Pitch Angle", "Right Pitch Angle", "Left Roll Angle", "Right Roll Angle"])
                                    
                    A_Time_diff = dt.datetime.today().timestamp()-Start_UpA
                    OR = convert(origin_Raw)
                    LS = convert(L_Shoulder)
                    RS = convert(R_Shoulder)
                    LH = convert(L_Hip)
                    RH = convert(R_Hip)
                    LK = convert(L_Knee)
                    RK = convert(R_Knee)
                    LA = convert(L_Ankle)
                    RA = convert(R_Ankle)
                    writerS.writerow([A_Time_diff, OR[0], OR[1], OR[2], LS[0], LS[1], LS[2], RS[0], RS[1], RS[2], LH[0], LH[1], LH[2], RH[0], RH[1], RH[2], LK[0], LK[1], LK[2], RK[0], RK[1], RK[2], LA[0], LA[1], LA[2], RA[0], RA[1], RA[2], L_Knee_Angle, R_Knee_Angle, L_yaw, R_yaw, L_pitch, R_pitch, L_roll, R_roll])
           
            elif key & 0xFF == ord('d'):
                if quality_check == 1:
                    if pre_e == False:
                        pre_e = True
                        Start_UpE = dt.datetime.today().timestamp()
                        f = open('/home/joey/Documents/python/BridgeStand/Record/240266/Down02.csv', 'w')
                        writerE = csv.writer(f)
                        writerE.writerow(["Time", "Origin_X", "Origin_Y", "Origin_Z", "Left_Shoulder_X", "Left_Shoulder_Y", "Left_Shoulder_Z", "Right_Shoulder_X", "Right_Shoulder_Y", "Right_Shoulder_Z", "Left_Hip_X", "Left_Hip_Y", "Left_Hip_Z", "Right_Hip_X", "Right_Hip_Y", "Right_Hip_Z", "Left_Knee_X", "Left_Knee_Y", "Left_Knee_Z", "Right_Knee_X", "Right_Knee_Y", "Right_Knee_Z", "Left_Ankle_X", "Left_Ankle_Y", "Left_Ankle_Z", "Right_Ankle_X", "Right_Ankle_Y", "Right_Ankle_Z", "Left Knee Angle", "Right Knee Angle", "Left Yaw Angle", "Right Yaw Angle", "Left Pitch Angle", "Right Pitch Angle", "Left Roll Angle", "Right Roll Angle"])
                                    
                    A_Time_diff = dt.datetime.today().timestamp()-Start_UpA
                    OR = convert(origin_Raw)
                    LS = convert(L_Shoulder)
                    RS = convert(R_Shoulder)
                    LH = convert(L_Hip)
                    RH = convert(R_Hip)
                    LK = convert(L_Knee)
                    RK = convert(R_Knee)
                    LA = convert(L_Ankle)
                    RA = convert(R_Ankle)
                    writerE.writerow([A_Time_diff, OR[0], OR[1], OR[2], LS[0], LS[1], LS[2], RS[0], RS[1], RS[2], LH[0], LH[1], LH[2], RH[0], RH[1], RH[2], LK[0], LK[1], LK[2], RK[0], RK[1], RK[2], LA[0], LA[1], LA[2], RA[0], RA[1], RA[2], L_Knee_Angle, R_Knee_Angle, L_yaw, R_yaw, L_pitch, R_pitch, L_roll, R_roll])
 
            else:
                pre_a = False
                pre_s = False
                pre_e = False

            



if __name__ == "__main__":
    Detect()
    
