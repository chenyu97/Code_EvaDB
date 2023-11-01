import time

start_time = time.time()

import evadb
import warnings
from IPython.core.display import HTML
from IPython import display


def pretty_print(df):
    return display(HTML(df.to_html().replace("\n","")))



cursor = evadb.connect().cursor()
warnings.filterwarnings("ignore")


cursor.query("DROP FUNCTION IF EXISTS Yolo;").df()
cursor.query("DROP FUNCTION IF EXISTS PropertyCalculate;").df()
cursor.query("DROP FUNCTION IF EXISTS NorFairTracker;").df()
cursor.query("DROP TABLE IF EXISTS ObjectDetectionVideos;").df()
cursor.query("LOAD VIDEO 'c001.mp4' INTO ObjectDetectionVideos;").df()

cursor.query("""
    CREATE FUNCTION Yolo
    IMPL './yolo_object_detector.py'
""").df()

cursor.query("""
    CREATE FUNCTION NorFairTracker
    IMPL './nor_fair.py'
""").df()

cursor.query("""
    CREATE FUNCTION PropertyCalculate
    IMPL './property_calculate.py'
""").df()

my_query = cursor.query("""           
        SELECT track_ids, validation, property
        FROM (SELECT id, track_ids, track_labels, track_bboxes, track_scores, 
                                PropertyCalculate(id, track_ids, track_labels, track_bboxes, track_scores)
              FROM (SELECT id, NorFairTracker(id, data, labels, bboxes, scores)
                    FROM (SELECT id, data, Yolo(data) 
                          FROM ObjectDetectionVideos) 
                    AS DET(id, data, labels, bboxes, scores)) 
              AS TRA(id, track_ids, track_labels, track_bboxes, track_scores)) 
        AS PRO(id, track_ids, track_labels, track_bboxes, track_scores, validation, property) WHERE [1] <@ validation
""")

response = my_query.df()
#print(response)
#print(type(response))
response.to_csv('property_speeding_car.csv', index=False)  # 如果您不希望导出索引，则设置index=False


cursor.query("DROP FUNCTION IF EXISTS Yolo;").df()
cursor.query("DROP FUNCTION IF EXISTS PropertyCalculate;").df()
cursor.query("DROP FUNCTION IF EXISTS NorFairTracker;").df()
cursor.query("DROP TABLE IF EXISTS ObjectDetectionVideos;").df()


end_time = time.time()

print('total cost: ' + str(end_time - start_time))