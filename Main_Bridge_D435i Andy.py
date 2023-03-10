
import pyrealsense2 as rs
import mediapipe as mp
import numpy as np
import cv2
import datetime as dt
import csv


def Vn(v1, v2):
  Vn = np.cross(v1, v2)
  Vn /= np.linalg.norm(Vn)
  return Vn

def getAng(vref, v, vaxis = None):
  if vaxis is None:
    n = Vn(vref, v)
  else:
    n = vaxis
  ang = np.arctan2(np.dot(np.cross(vref, v), n), np.dot(vref, v))
  return ang

def getProj(v, vn):
  vProj = v - np.dot(v, vn) * vn
  return vProj

def convert(array):
  result = {"x": array[0],
            "y": array[1],
            "z": array[2]
            }
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
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=True,
        min_detection_confidence=0.8) as pose:
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

    
    # For webcam input:
    with mp_pose.Pose(
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8) as pose:

        # a = 0
        # s = 0
        # d = 0
        pre_a = False
        pre_s = False
        pre_d = False

        while True:
            start_time = dt.datetime.today().timestamp() # Necessary for FPS calculations

            # Get and align frames
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            # Accel_frame = frames[2].as_motion_frame().get_motion_data()

            # accel = [Accel_frame.x, Accel_frame.y, Accel_frame.z]
            # Cam_Ang_X = -np.deg2rad(90-np.rad2deg(np.arccos(np.dot(np.array([0,1]),np.array([accel[1],accel[2]]))/(np.linalg.norm(np.array([accel[1],accel[2]]))))))
            # R_Matrix = np.linalg.inv(np.array(
            #     [[1,               0,                0],
            #     [0, np.cos(Cam_Ang_X),  np.sin(Cam_Ang_X)],
            #     [0, -np.sin(Cam_Ang_X), np.cos(Cam_Ang_X)]]
            # ))


            # if not aligned_depth_frame or not color_frame:
            #     continue

            # Process images
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            depth_image_flipped = cv2.flip(depth_image,1)
            color_image = np.asanyarray(color_frame.get_data())

            depth_image_3d = np.dstack((depth_image,depth_image,depth_image))
            #background_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), background_removed_color, color_image)

            #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            #image = cv2.flip(background_removed,1)
            image = cv2.flip(color_image,1)

            color_image = cv2.flip(color_image,1)
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
                L_Shoulder_y = (stream_res_y/2-L_Shoulder_y)*2*L_Shoulder_z*np.tan(np.deg2rad(42/2))/stream_res_y
                L_Shoulder_x = (L_Shoulder_x-stream_res_x/2)*2*L_Shoulder_z*np.tan(np.deg2rad(69/2))/stream_res_x
                
                L_Shoulder = np.array([L_Shoulder_x, L_Shoulder_y, L_Shoulder_z])
                # L_Shoulder_raw = (np.matmul(R_Matrix, np.array([L_Shoulder_x, L_Shoulder_y, L_Shoulder_z]))).tolist()
                #print("L_Shoulder_Pos", L_Shoulder_raw)


                #Right Shoulder
                R_Shoulder_position = results.pose_landmarks.landmark[12]
                R_Shoulder_x = R_Shoulder_position.x*len(depth_image_flipped[12])
                R_Shoulder_y = R_Shoulder_position.y*len(depth_image_flipped)
                if R_Shoulder_x >= len(depth_image_flipped[12]):
                    R_Shoulder_x = len(depth_image_flipped[12]) - 1
                if R_Shoulder_y >= len(depth_image_flipped):
                    R_Shoulder_y = len(depth_image_flipped) - 1
                
                R_Shoulder_z = depth_image_flipped[int(R_Shoulder_y),int(R_Shoulder_x)] * depth_scale *1000 # meters
                R_Shoulder_y = (stream_res_y/2-R_Shoulder_y)*2*R_Shoulder_z*np.tan(np.deg2rad(42/2))/stream_res_y
                R_Shoulder_x = (R_Shoulder_x-stream_res_x/2)*2*R_Shoulder_z*np.tan(np.deg2rad(69/2))/stream_res_x
                
                R_Shoulder = np.array([R_Shoulder_x, R_Shoulder_y, R_Shoulder_z])
                # R_Shoulder_raw = (np.matmul(R_Matrix, np.array([R_Shoulder_x, R_Shoulder_y, R_Shoulder_z]))).tolist()
                #print("R_Shoulder_Pos", R_Shoulder_raw)
        

                #Left_Hip
                L_Hip_position = results.pose_landmarks.landmark[23]
                L_Hip_x = L_Hip_position.x*len(depth_image_flipped[23])
                L_Hip_y = L_Hip_position.y*len(depth_image_flipped)
                if L_Hip_x >= len(depth_image_flipped[23]):
                    L_Hip_x = len(depth_image_flipped[23]) - 1
                if L_Hip_y >= len(depth_image_flipped):
                    L_Hip_y = len(depth_image_flipped) - 1
                
                L_Hip_z = depth_image_flipped[int(L_Hip_y),int(L_Hip_x)] * depth_scale *1000 # meters
                L_Hip_y = (stream_res_y/2-L_Hip_y)*2*L_Hip_z*np.tan(np.deg2rad(42/2))/stream_res_y
                L_Hip_x = (L_Hip_x-stream_res_x/2)*2*L_Hip_z*np.tan(np.deg2rad(69/2))/stream_res_x
                
                L_Hip = np.array([L_Hip_x, L_Hip_y, L_Hip_z])
                # L_Hip_raw = (np.matmul(R_Matrix, np.array([L_Hip_x, L_Hip_y, L_Hip_z]))).tolist()
                #print("L_Hip_Pos", L_Hip_raw)



                #Right_Hip
                R_Hip_position = results.pose_landmarks.landmark[24]
                R_Hip_x = R_Hip_position.x*len(depth_image_flipped[24])
                R_Hip_y = R_Hip_position.y*len(depth_image_flipped)
                if R_Hip_x >= len(depth_image_flipped[24]):
                    R_Hip_x = len(depth_image_flipped[24]) - 1
                if R_Hip_y >= len(depth_image_flipped):
                    R_Hip_y = len(depth_image_flipped) - 1
                
                R_Hip_z = depth_image_flipped[int(R_Hip_y),int(R_Hip_x)] * depth_scale *1000 # meters
                R_Hip_y = (stream_res_y/2-R_Hip_y)*2*R_Hip_z*np.tan(np.deg2rad(42/2))/stream_res_y
                R_Hip_x = (R_Hip_x-stream_res_x/2)*2*R_Hip_z*np.tan(np.deg2rad(69/2))/stream_res_x
                
                R_Hip = np.array([R_Hip_x, R_Hip_y, R_Hip_z])
                # R_Hip_raw = (np.matmul(R_Matrix, np.array([R_Hip_x, R_Hip_y, R_Hip_z]))).tolist()
                #print("R_Hip_Pos", R_Hip_raw)


                #Left_Knee
                L_Knee_position = results.pose_landmarks.landmark[25]
                L_Knee_x = L_Knee_position.x*len(depth_image_flipped[25])
                L_Knee_y = L_Knee_position.y*len(depth_image_flipped)
                if L_Knee_x >= len(depth_image_flipped[25]):
                    L_Knee_x = len(depth_image_flipped[25]) - 1
                if L_Knee_y >= len(depth_image_flipped):
                    L_Knee_y = len(depth_image_flipped) - 1               
                
                L_Knee_z = depth_image_flipped[int(L_Knee_y),int(L_Knee_x)] * depth_scale *1000 # meters
                L_Knee_y = (stream_res_y/2-L_Knee_y)*2*L_Knee_z*np.tan(np.deg2rad(42/2))/stream_res_y
                L_Knee_x = (L_Knee_x-stream_res_x/2)*2*L_Knee_z*np.tan(np.deg2rad(69/2))/stream_res_x
                
                L_Knee = np.array([L_Knee_x, L_Knee_y, L_Knee_z])
                # L_Knee_raw = (np.matmul(R_Matrix, np.array([L_Knee_x, L_Knee_y, L_Knee_z]))).tolist()
                #print("L_Knee_Pos", L_Knee_raw)



                #Right_Knee
                R_Knee_position = results.pose_landmarks.landmark[26]
                R_Knee_x = R_Knee_position.x*len(depth_image_flipped[26])
                R_Knee_y = R_Knee_position.y*len(depth_image_flipped)
                if R_Knee_x >= len(depth_image_flipped[26]):
                    R_Knee_x = len(depth_image_flipped[26]) - 1
                if R_Knee_y >= len(depth_image_flipped):
                    R_Knee_y = len(depth_image_flipped) - 1
                
                R_Knee_z = depth_image_flipped[int(R_Knee_y),int(R_Knee_x)] * depth_scale *1000 # meters
                R_Knee_y = (stream_res_y/2-R_Knee_y)*2*R_Knee_z*np.tan(np.deg2rad(42/2))/stream_res_y
                R_Knee_x = (R_Knee_x-stream_res_x/2)*2*R_Knee_z*np.tan(np.deg2rad(69/2))/stream_res_x
                
                R_Knee = np.array([R_Knee_x, R_Knee_y, R_Knee_z])
                # R_Knee_raw = (np.matmul(R_Matrix, np.array([R_Knee_x, R_Knee_y, R_Knee_z]))).tolist()
                #print("R_Knee_Pos", R_Knee_raw)
                
               


                #Left_Ankle
                L_Ankle_position = results.pose_landmarks.landmark[27]
                L_Ankle_x = L_Ankle_position.x*len(depth_image_flipped[27])
                L_Ankle_y = L_Ankle_position.y*len(depth_image_flipped)
                if L_Ankle_x >= len(depth_image_flipped[23]):
                    L_Ankle_x = len(depth_image_flipped[23]) - 1
                if L_Ankle_y >= len(depth_image_flipped):
                    L_Ankle_y = len(depth_image_flipped) - 1
                
                L_Ankle_z = depth_image_flipped[int(L_Ankle_y),int(L_Ankle_x)] * depth_scale *1000 # meters
                L_Ankle_y = (stream_res_y/2-L_Ankle_y)*2*L_Ankle_z*np.tan(np.deg2rad(42/2))/stream_res_y
                L_Ankle_x = (L_Ankle_x-stream_res_x/2)*2*L_Ankle_z*np.tan(np.deg2rad(69/2))/stream_res_x
                
                L_Ankle = np.array([L_Ankle_x, L_Ankle_y, L_Ankle_z])
                # L_Ankle_raw = (np.matmul(R_Matrix, np.array([L_Ankle_x, L_Ankle_y, L_Ankle_z]))).tolist()
                #print("L_Ankle_Pos", L_Ankle_raw)



                #Right_Ankle
                R_Ankle_position = results.pose_landmarks.landmark[28]
                R_Ankle_x = R_Ankle_position.x*len(depth_image_flipped[28])
                R_Ankle_y = R_Ankle_position.y*len(depth_image_flipped)
                if R_Ankle_x >= len(depth_image_flipped[24]):
                    R_Ankle_x = len(depth_image_flipped[24]) - 1
                if R_Ankle_y >= len(depth_image_flipped):
                    R_Ankle_y = len(depth_image_flipped) - 1

                R_Ankle_z = depth_image_flipped[int(R_Ankle_y),int(R_Ankle_x)] * depth_scale *1000 # meters
                R_Ankle_y = (stream_res_y/2-R_Ankle_y)*2*R_Ankle_z*np.tan(np.deg2rad(42/2))/stream_res_y
                R_Ankle_x = (R_Ankle_x-stream_res_x/2)*2*R_Ankle_z*np.tan(np.deg2rad(69/2))/stream_res_x
                
                R_Ankle = np.array([R_Ankle_x, R_Ankle_y, R_Ankle_z])
                # R_Ankle_raw = (np.matmul(R_Matrix, np.array([R_Ankle_x, R_Ankle_y, R_Ankle_z]))).tolist()
                #print("R_Ankle_Pos", R_Ankle_raw)


                # origin_Raw = [(L_Shoulder_raw[0]+R_Shoulder_raw[0])/2, (L_Shoulder_raw[1]+R_Shoulder_raw[1])/2, (L_Shoulder_raw[2]+R_Shoulder_raw[2])/2]
                #Tmatrix = np.linalg.inv(np.array(
                 #       [[0,   -1,  0, origin_Raw[0]],
                  #      [ 0,    0, -1, origin_Raw[1]],
                   #     [-1,    0,  0, origin_Raw[2]],
                    #    [0,     0,  0,            1]]))

                #L_Shoulder_raw.append(1)
                #R_Shoulder_raw.append(1)
                #L_Hip_raw.append(1)
                #R_Hip_raw.append(1)
                #L_Knee_raw.append(1)
                #R_Knee_raw.append(1)
                #L_Ankle_raw.append(1)
                #R_Ankle_raw.append(1)


                # L_Shoulder = [L_Shoulder_raw[0]-origin_Raw[0], L_Shoulder_raw[1]-origin_Raw[1], L_Shoulder_raw[2]- origin_Raw[2]]
                # R_Shoulder = [R_Shoulder_raw[0]-origin_Raw[0], R_Shoulder_raw[1]-origin_Raw[1], R_Shoulder_raw[2]- origin_Raw[2]]
                # L_Hip = [L_Hip_raw[0]-origin_Raw[0], L_Hip_raw[1]-origin_Raw[1], L_Hip_raw[2]- origin_Raw[2]]
                # R_Hip = [R_Hip_raw[0]-origin_Raw[0], R_Hip_raw[1]-origin_Raw[1], R_Hip_raw[2]- origin_Raw[2]]
                # L_Knee = [L_Knee_raw[0]-origin_Raw[0], L_Knee_raw[1]-origin_Raw[1], L_Knee_raw[2]- origin_Raw[2]]
                # R_Knee = [R_Knee_raw[0]-origin_Raw[0], R_Knee_raw[1]-origin_Raw[1], R_Knee_raw[2]- origin_Raw[2]]
                # L_Ankle = [L_Ankle_raw[0]-origin_Raw[0], L_Ankle_raw[1]-origin_Raw[1], L_Ankle_raw[2]- origin_Raw[2]]
                # R_Ankle = [R_Ankle_raw[0]-origin_Raw[0], R_Ankle_raw[1]-origin_Raw[1], R_Ankle_raw[2]- origin_Raw[2]]

                
                #L_Shoulder = np.matmul(Tmatrix, np.array(L_Shoulder_raw))
                #R_Shoulder = np.matmul(Tmatrix, np.array(R_Shoulder_raw))
                #L_Hip = np.matmul(Tmatrix, np.array(L_Hip_raw))
                #R_Hip = np.matmul(Tmatrix, np.array(R_Hip_raw))
                #L_Knee = np.matmul(Tmatrix, np.array(L_Knee_raw))
                #R_Knee = np.matmul(Tmatrix, np.array(R_Knee_raw))
                #L_Ankle = np.matmul(Tmatrix, np.array(L_Ankle_raw))
                #R_Ankle = np.matmul(Tmatrix, np.array(R_Ankle_raw))

                ref = (L_Shoulder + R_Shoulder) / 2

                L_Shoulder -= ref
                R_Shoulder -= ref
                L_Hip -= ref
                R_Hip -= ref
                L_Knee -= ref
                R_Knee -= ref
                L_Ankle -= ref
                R_Ankle -= ref

                Body = Vn(R_Hip, L_Hip)

                R_Thigh = R_Knee - R_Hip
                R_Proj = getProj(R_Thigh, Body)
                hip = L_Hip - R_Hip
                R_Yaw = getAng(R_Proj, hip, Body)
                R_Pitch = getAng(R_Thigh, R_Proj)
                R_Calf = R_Ankle - R_Knee
                R_Thigh *= -1
                R_Plane = Vn(R_Calf, R_Thigh)
                R_Roll = getAng(Body, R_Plane)
                R_Theta = getAng(R_Calf, R_Thigh, R_Plane)


                L_Thigh = L_Knee - L_Hip
                L_Proj = getProj(L_Thigh, Body)
                hip *= -1
                L_Yaw = getAng(L_Proj, hip, Body)
                L_Pitch = getAng(L_Thigh, L_Proj)
                L_Calf = L_Ankle - L_Knee
                L_Thigh *= -1
                L_Plane = Vn(L_Calf, L_Thigh)
                L_Roll = getAng(L_Plane, Body)
                L_Theta = getAng(L_Calf, L_Thigh, L_Plane)

                print("Left Shoulder", L_Shoulder[0:3])
                print("Right Shoulder", R_Shoulder[0:3])
                print("Left Hip", L_Hip[0:3])
                print("Right Hip", R_Hip[0:3])
                print("Left Knee", L_Knee[0:3])
                print("Right Knee", R_Knee[0:3])
                print("Left Ankle", L_Ankle[0:3])
                print("Right Ankle", R_Ankle[0:3])
                print("Right Yaw", np.rad2deg(R_Yaw))
                print("Right Pitch", np.rad2deg(R_Pitch))
                print("Right Roll", np.rad2deg(R_Roll))
                print("Right Theta", np.rad2deg(R_Theta))
                
                key = cv2.waitKey(1)
                # Press esc or 'q' to close the image window
                if key & 0xFF == ord('q') or key == 27:
                    print(f"User pressed break key for SN: {device}")
                    print(f"Exiting loop for SN: {device}")
                    print(f"Application Closing.")
                    pipeline.stop()
                    print(f"Application Closed.")
                    break

                if key & 0xFF == ord('a'):
                    if pre_a == False:
                        pre_a = True
                        Start_Up = dt.datetime.today().timestamp()
                        fA = open('/home/joey/Documents/python/BridgeStand/Record/Correct/Up02.csv', 'w')
                        writerA = csv.writer(fA)
                        # writerA.writerow(["Time", "Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"])

                    A_Time_diff = dt.datetime.today().timestamp()-Start_Up
                    writerA.writerow({
                        "Time": A_Time_diff,
                        "Ref": convert(ref),
                        "Body": convert(Body),
                        
                        "R_Shoulder": convert(R_Shoulder),
                        "R_Hip": convert(R_Hip),
                        "R_Knee": convert(R_Knee),
                        "R_Ankle": convert(R_Ankle),
                        "R_Yaw": R_Yaw,
                        "R_Pitch": R_Pitch,
                        "R_Roll": R_Roll,
                        "R_Theta": R_Theta,
                                                
                        "L_Shoulder": convert(L_Shoulder),
                        "L_Hip": convert(L_Hip),
                        "L_Knee": convert(L_Knee),
                        "L_Ankle": convert(L_Ankle),
                        "L_Yaw": L_Yaw,
                        "L_Pitch": L_Pitch,
                        "L_Roll": L_Roll,
                        "L_Theta": L_Theta,

                        "Label": "up"
                    })
                else:
                    pre_a = False

                if key & 0xFF == ord('s'):
                    if pre_s == False:
                        pre_s = True
                        Start_Hold = dt.datetime.today().timestamp()
                        fS = open('/home/joey/Documents/python/BridgeStand/Record/Correct/Hold02.csv', 'w')
                        writerS = csv.writer(fS)
                        # writerS.writerow(["Time", "Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"])
                    
                    S_Time_diff = dt.datetime.today().timestamp()-Start_Hold
                    writerS.writerow({
                        "Time": S_Time_diff,
                        "Ref": convert(ref),
                        "Body": convert(Body),
                        
                        "R_Shoulder": convert(R_Shoulder),
                        "R_Hip": convert(R_Hip),
                        "R_Knee": convert(R_Knee),
                        "R_Ankle": convert(R_Ankle),
                        "R_Yaw": R_Yaw,
                        "R_Pitch": R_Pitch,
                        "R_Roll": R_Roll,
                        "R_Theta": R_Theta,
                                                
                        "L_Shoulder": convert(L_Shoulder),
                        "L_Hip": convert(L_Hip),
                        "L_Knee": convert(L_Knee),
                        "L_Ankle": convert(L_Ankle),
                        "L_Yaw": L_Yaw,
                        "L_Pitch": L_Pitch,
                        "L_Roll": L_Roll,
                        "L_Theta": L_Theta,

                        "Label": "hold_Correct"
                    })
                else:
                    pre_s = False

                if key & 0xFF == ord('d'):
                    if pre_d == False:
                        pre_d = True
                        Start_Down = dt.datetime.today().timestamp()
                        fD = open('/home/joey/Documents/python/BridgeStand/Record/Correct/Down02.csv', 'w')
                        writerD = csv.writer(fD)
                        # writerD.writerow(["Time", "Left Shoulder", "Right Shoulder", "Left Hip", "Right Hip", "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"])

                    D_Time_diff = dt.datetime.today().timestamp()-Start_Down
                    writerD.writerow({
                        "Time": D_Time_diff,
                        "Ref": convert(ref),
                        "Body": convert(Body),
                        
                        "R_Shoulder": convert(R_Shoulder),
                        "R_Hip": convert(R_Hip),
                        "R_Knee": convert(R_Knee),
                        "R_Ankle": convert(R_Ankle),
                        "R_Yaw": R_Yaw,
                        "R_Pitch": R_Pitch,
                        "R_Roll": R_Roll,
                        "R_Theta": R_Theta,
                                                
                        "L_Shoulder": convert(L_Shoulder),
                        "L_Hip": convert(L_Hip),
                        "L_Knee": convert(L_Knee),
                        "L_Ankle": convert(L_Ankle),
                        "L_Yaw": L_Yaw,
                        "L_Pitch": L_Pitch,
                        "L_Roll": L_Roll,
                        "L_Theta": L_Theta,

                        "Label": "down"
                    })
                else:
                    pre_d = False

                # image = cv2.flip(image,1)
                image = cv2.putText(image, f"Body Detected", org, font, fontScale, color, thickness, cv2.LINE_AA)
 
            else:
                # image = cv2.flip(image,1)
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

if __name__ == "__main__":
    Detect()
    
    