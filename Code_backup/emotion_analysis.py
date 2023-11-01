import evadb
import warnings
import cv2
from pprint import pprint
from matplotlib import pyplot as plt
from ipywidgets import Video, Image


def annotate_video(detections, input_video_path, output_video_path):
    color1=(207, 248, 64)
    color2=(255, 49, 49)
    thickness=4

    vcap = cv2.VideoCapture(input_video_path)
    width = int(vcap.get(3))
    height = int(vcap.get(4))
    fps = vcap.get(5)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') #codec
    video=cv2.VideoWriter(output_video_path, fourcc, fps, (width,height))

    frame_id = 0
    # Capture frame-by-frame
    # ret = 1 if the video is captured; frame is the image
    ret, frame = vcap.read() 

    while ret:
        df = detections
        df = df[['Face.bbox', 'emotiondetector.labels', 'emotiondetector.scores']][df.index == frame_id]
        if df.size:
            
            x1, y1, x2, y2 = df['Face.bbox'].values[0]
            label = df['emotiondetector.labels'].values[0]
            score = df['emotiondetector.scores'].values[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # object bbox
            frame=cv2.rectangle(frame, (x1, y1), (x2, y2), color1, thickness) 
            # object label
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color1, thickness)
            # object score
            cv2.putText(frame, str(round(score, 5)), (x1+120, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color1, thickness)
            # frame label
            cv2.putText(frame, 'Frame ID: ' + str(frame_id), (700, 500), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color2, thickness) 
        
            video.write(frame)
            # Show every fifth frame
            if frame_id % 5 == 0:
                plt.imshow(frame)
                plt.show()
        
        frame_id+=1
        ret, frame = vcap.read()

    video.release()
    vcap.release()


cursor = evadb.connect().cursor()
warnings.filterwarnings("ignore")

#print(cursor.query("SHOW FUNCTIONS;").df())
#cursor.query("DROP FUNCTION EmotionDetector").df()
cursor.query("DROP FUNCTION IF EXISTS FaceDetector;").df()
cursor.query("DROP FUNCTION IF EXISTS EmotionDetector;").df()

cursor.query("DROP TABLE IF EXISTS HAPPY;").df()
cursor.query("LOAD VIDEO 'defhappy.mp4' INTO HAPPY").df()
cursor.query("""
    CREATE FUNCTION IF NOT EXISTS EmotionDetector 
    INPUT (frame NDARRAY UINT8(3, ANYDIM, ANYDIM)) 
    OUTPUT (labels NDARRAY STR(ANYDIM), scores NDARRAY FLOAT32(ANYDIM)) 
    TYPE  Classification IMPL 'emotion_detector.py';
""").df()
cursor.query("""
    CREATE FUNCTION IF NOT EXISTS FaceDetector
    INPUT  (frame NDARRAY UINT8(3, ANYDIM, ANYDIM))
    OUTPUT (bboxes NDARRAY FLOAT32(ANYDIM, 4),
          scores NDARRAY FLOAT32(ANYDIM))
    TYPE  FaceDetection
    IMPL  'face_detector.py';
""").df()
#print(cursor.query("SHOW FUNCTIONS;").df())
print(cursor.query("SELECT id, FaceDetector(data) FROM HAPPY WHERE id < 16").df())
query = cursor.query("""
    SELECT id, bbox, EmotionDetector(Crop(data, bbox))
    FROM HAPPY JOIN LATERAL UNNEST(FaceDetector(data)) AS Face(bbox, conf)
    WHERE id < 15
""")
response = query.df()
print(response)

input_path = 'defhappy.mp4'
output_path = 'video.mp4'

annotate_video(response, input_path, output_path)

#cursor.query("DROP FUNCTION EmotionDetector").df()
#cursor.query("DROP FUNCTION FaceDetector").df()
#print(cursor.query("SHOW FUNCTIONS;").df())