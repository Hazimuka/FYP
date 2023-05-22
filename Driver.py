import cv2
import numpy as np
import math
import time as tm
import matplotlib.pyplot as plt
import os
from gaze_tracking import GazeTracking

flag_debug = False
start_time= 0
end_time = 0
gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
if not os.path.exists('exported_images'):
    os.makedirs('exported_images')

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

inWidth = 368
inHeight = 368

try:
    net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
except Exception as e:
     print("**ERROR**\nAn error Occured while Opening graph_opt.pb")
     exit()
time_series=[]
cheating = 0
time = 0
not_detected= 0
flag_timer = False
lapsed_flag=False
while True:
        # We get a new frame from the webcam
    _, frame = webcam.read()
    
    try:
        gaze.refresh(frame)
    except Exception as e:
        print("**ERROR**\nGaze Tracking has encounted a fatal Error")
        exit()
    frame = gaze.annotated_frame()
    text = ""
    eye_var = gaze.pupils_located
    if eye_var == True:
        if gaze.is_center():#and not gaze.is_bottom() and not gaze.is_top():
            if gaze.is_top():
                text = "Looking Top"
            elif gaze.is_bottom():
                text = "Looking Bottom"
            else:
                text = "Looking Center"
        elif gaze.is_bottom() and gaze.is_right():
                text = "Looking Bottom Right"
        elif gaze.is_bottom() and gaze.is_left():
                text = "Looking Bottom Left"
        elif gaze.is_bottom():
                text = "Looking Bottom SUS"
        elif gaze.is_top() and gaze.is_right():
                text = "Looking Top Right"
        elif gaze.is_top() and gaze.is_left():
                text = "Looking Top Left"
        elif gaze.is_top():
                text = "Looking Top"            
        elif gaze.is_right():
                text = "Looking Right"
        elif gaze.is_left():
                text = "Looking Left"
    else:
        text = "**ERROR** EYES NOT VISIBLE!"
        
       
        
    if flag_debug:
        cv2.putText(frame, text, (90,60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147,58,31), 2)
    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    vert_ratio = gaze.vertical_ratio()
    horz_ratio = gaze.horizontal_ratio()

    if flag_debug:
         cv2.putText(frame ,"Vertical Ratio " + str(vert_ratio), (10 ,200), cv2.FONT_HERSHEY_DUPLEX, 0.95, (147,58,31), 1)
         cv2.putText(frame ,"Horizontal Ratio " + str(horz_ratio), (10, 235), cv2.FONT_HERSHEY_DUPLEX, 0.95, (147,58,31), 1)
    #########################################################
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    try:
        net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    except Exception as e:
        print("**ERROR**\n Fatal Issue with mapping DNN to net variable")
        exit()
    out = net.forward()
    out = out[:, :19, :, :]
    try:
        assert(len(BODY_PARTS) == out.shape[1])
     
        points = []
        for i in range(len(BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > 0.05 else None)

        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in BODY_PARTS)
            assert(partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]

            if points[idFrom] and points[idTo] and flag_debug:
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
    except Exception as e:
        print("**ERROR**\n Issues with asserting bodybody parts!")
        exit()
    def angle_between_points(point1, point2):
        if (point1 != None and point2 != None):
            x1, y1 = point1
            x2, y2 = point2
            return math.degrees(math.atan2(y2 - y1, x2 - x1))
        else:
            return 0
    t, _ = net.getPerfProfile()
    if flag_debug:
         cv2.putText(frame, str(angle_between_points(points[5],points[6])), (90,120), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147,58,31), 2)
    behave= "Not started"
    if (not gaze.pupils_located or points[2] == None or points[3] == None or points[5] == None or points[6] == None):
        time_series.append(0)
        if(not gaze.pupils_located):
            behave = "Eyes not detected"
        elif(points[3] == None):
            behave = "Right elbow not detected"
        elif(points[6] == None):
            behave = "Left Elbow not detected"
        not_detected =not_detected +1;
        start_time = 0
        flag_timer = False
        end_time = 0

    elif angle_between_points(points[2],points[3]) > 160 or angle_between_points(points[2],points[3])<90 or angle_between_points(points[5],points[6]) < 30 or angle_between_points(points[5],points[6]) > 90 and time >30:
        if lapsed_flag==True:
            cheating= cheating + 1
            time_series.append(2)
        start_time = 0
        flag_timer = False
        end_time = 0
        behave = "Abnormal body"
        filename = f"exported_images/{len(os.listdir('exported_images')) + 1}.jpg"
        cv2.imwrite(filename, frame)

    elif (gaze.is_left() and not gaze.is_top() ) or (gaze.is_right() and not gaze.is_top() and time > 30):
        if lapsed_flag==True:
            cheating= cheating + 1
            time_series.append(2) 
        start_time = 0
        flag_timer = False
        end_time= 0 
        behave = "Abnormal eyes"
        filename = f"exported_images/{len(os.listdir('exported_images')) + 1}.jpg"
        cv2.imwrite(filename, frame)

    else:
        if flag_timer==False:
            start_time = tm.time()
            flag_timer = True
            
        behave = "Normal"
        
        if lapsed_flag==True:
            time_series.append(1)
            
    end_time = tm.time()

    if(flag_timer == False):
        time_lapsed=0
    else:
        time_lapsed = end_time - start_time
     
    if time_lapsed>5:
        lapsed_flag=True
    cv2.putText(frame, behave, (90,180), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147,58,31), 2)
    if flag_debug:
        cv2.putText(frame,"Timer : " + str(time_lapsed%60), (90,300), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147,58,31), 2)
    cv2.putText(frame,"Inference Time: {:.2f} ms".format(t * 1000.0 / cv2.getTickFrequency()), (90,350), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147,58,31), 2)
    cv2.imshow('OpenPose using OpenCV', frame)
    key = cv2.waitKey(5)
    if key == 27:
        flag_debug = not flag_debug;
    if key == 113:
        break
    time = time + 1
    

webcam.release()
cv2.destroyAllWindows()
time_values = np.arange(len(time_series))
plt.plot(time_values, time_series)
plt.title("Time Series Graph")
plt.xlabel("Time")
plt.ylabel("Value")
plt.show()
percent = cheating/time *100
p_not_detected = not_detected/time *100
print( "Cheating Percentage = " + str(percent))
print("Not detected Percentage = " + str(p_not_detected))