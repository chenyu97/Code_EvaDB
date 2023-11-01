import evadb
import warnings
from IPython.core.display import HTML
from IPython import display
import time


def pretty_print(df):
    return display(HTML(df.to_html().replace("\n","")))

cursor = evadb.connect().cursor()
warnings.filterwarnings("ignore")

cursor.query("DROP TABLE IF EXISTS MyVideo;").df()

load_video_query = """
                LOAD VIDEO 'ua_detrac.mp4' INTO MyVideo;
                """
cursor.query(load_video_query).df()

select_query = """
            SELECT id, T.iids, T.bboxes, T.scores, T.labels
            FROM MyVideo JOIN LATERAL EXTRACT_OBJECT(data, Yolo, NorFairTracker)
                AS T(iids, labels, bboxes, scores)
            WHERE id < 30;
            """
#response = cursor.query(select_query).df()
#print(response)
#response.to_csv('select_query.csv', index=False)


select_query = """           
              SELECT id, track_ids, track_bboxes, track_scores, track_labels
              FROM (SELECT id, NorFairTracker(id, data, labels, bboxes, scores)
                    FROM (SELECT id, data, Yolo(data) 
                          FROM MyVideo WHERE id < 30) 
                    AS DET(id, data, labels, bboxes, scores)) 
              AS TRA(id, track_ids, track_labels, track_bboxes, track_scores)

"""
#response = cursor.query(select_query).df()
#print(response)
#response.to_csv('select_query_simple.csv', index=False)


select_query = """
            SELECT id, T.iid, T.bbox, T.score, T.label
            FROM MyVideo JOIN LATERAL
                UNNEST(EXTRACT_OBJECT(data, Yolo, NorFairTracker)) AS T(iid, label, bbox, score)
            WHERE id < 30;
            """
#response = cursor.query(select_query).df()
#print(response)
#response.to_csv('select_query_unnest.csv', index=False)


'''
cursor.query("DROP FUNCTION IF EXISTS YoloTest;").df()

cursor.query("""
    CREATE FUNCTION YoloTest
    IMPL './yolo_test.py'
""").df()

cursor.query("DROP FUNCTION IF EXISTS YoloTest2;").df()

cursor.query("""
    CREATE FUNCTION YoloTest2
    IMPL './yolo_test2.py'
""").df()
'''

select_query = """
            SELECT id, data 
            FROM MyVideo WHERE ['plane'] <@ YoloTest2(data).labels AND ['plane'] <@ YoloTest(data).labels
            """
#response = cursor.query(select_query).df()
#print(response)


select_query = """
            SELECT *
            FROM MyVideo
            """
#response = cursor.query(select_query).df()
#print(response)

select_query = """
            SELECT Yolo(data)
            FROM MyVideo
            """
#response = cursor.query(select_query).df()
#print(response)

select_query = """
            SELECT *
            FROM MyVideo JOIN LATERAL Yolo(data) AS yolo(labels, bboxes, scores)
            """
#response = cursor.query(select_query).df()
#print(response)

#cursor.query("DROP TABLE IF EXISTS MyTable;").df()

select_query = """
            CREATE TABLE MyTable_1 AS SELECT name, id, seconds FROM MyVideo
            """
#response = cursor.query(select_query).df()
#print(response)

select_query = """
            CREATE TABLE MyTable_2 AS SELECT name, id, data FROM MyVideo
            """
#response = cursor.query(select_query).df()
#print(response)

select_query = """
            SELECT * FROM MyTable_1
            """
#response = cursor.query(select_query).df()
#print(response)

select_query = """
            SELECT * FROM MyTable_2
            """
#response = cursor.query(select_query).df()
#print(response)

select_query = """
            SELECT * FROM MyTable_1 JOIN MyTable_2 ON mytable_1.name = mytable_2.name
            """
#response = cursor.query(select_query).df()
#print(response)

select_query = """
            SELECT * FROM MyTable_1 JOIN MyTable_2 ON mytable_1.name = mytable_2.name AND mytable_1.id = mytable_2.id
            """
#response = cursor.query(select_query).df()
#print(response)


cursor.query("DROP FUNCTION IF EXISTS Yolo;").df()

cursor.query("""
    CREATE FUNCTION Yolo
    IMPL './yolo_detector.py'
""").df()

query = """
            DROP FUNCTION IF EXISTS 
                NorFairTracker;
        """
cursor.query(query).df()

cursor.query("""
    CREATE FUNCTION NorFairTracker
    IMPL './nor_fair_tracker.py'
""").df()


select_query = """
            SELECT id, T.iid, T.bbox, T.score, T.label
            FROM MyVideo JOIN LATERAL
                UNNEST(EXTRACT_OBJECT(data, Yolo, NorFairTracker)) AS T(iid, label, bbox, score)
            WHERE T.label = 'car';
            """
response = cursor.query(select_query).df()
print(response)
response.to_csv('select_query_unnest.csv', index=False)


'''
select_query = """
            SELECT id, T.iids, T.bboxes, T.scores, T.labels
            FROM MyVideo JOIN LATERAL EXTRACT_OBJECT(data, Yolo, NorFairTracker)
                AS T(iids, labels, bboxes, scores)
            WHERE id < 10;
            """
response = cursor.query(select_query).df()
print(response)
response.to_csv('select_query.csv', index=False)
'''

'''
select_query = """               
              SELECT id, NorFairTracker(id, data, labels, bboxes, scores)
                    FROM (SELECT id, data, Yolo(data) 
                          FROM MyVideo WHERE id < 10) 
                    AS DET(id, data, labels, bboxes, scores)
"""
response = cursor.query(select_query).df()
print(response)
response.to_csv('select_query_simple1.csv', index=False)
'''

'''
select_query = """               
              SELECT id, NorFairTracker(id, data, labels, bboxes, scores)
                    FROM (SELECT id, data, Yolo(data) 
                          FROM MyVideo) 
                    AS DET(id, data, labels, bboxes, scores)
              WHERE id < 10
"""
response = cursor.query(select_query).df()
print(response)
response.to_csv('select_query_simple2.csv', index=False)
'''


select_query = """           
              SELECT id, track_ids, track_bboxes, track_scores, track_labels
              FROM (SELECT id, NorFairTracker(id, data, labels, bboxes, scores)
                    FROM (SELECT id, data, Yolo(data) 
                          FROM MyVideo ) 
                    AS DET(id, data, labels, bboxes, scores)) 
              AS TRA(id, track_ids, track_labels, track_bboxes, track_scores)
              WHERE id < 10

"""
#response = cursor.query(select_query).df()
#print(response)
#response.to_csv('select_query_simple3.csv', index=False)

'''
select_query = """
            SELECT 
                id, YOLO.bbox
            FROM 
                MyVideo
            JOIN LATERAL  
                UNNEST(Yolo(data)) 
            AS 
                YOLO(label, bbox, score)
            WHERE 
                YOLO.label = 'car' 
            AND 
                RecognizeColor(data, YOLO.bbox) = 'red';
"""
response = cursor.query(select_query).df()
print(response)
'''

cursor.query("DROP TABLE IF EXISTS MyVideo;").df()