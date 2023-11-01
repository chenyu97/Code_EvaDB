import evadb
import warnings
import cv2
from pprint import pprint
from matplotlib import pyplot as plt
from ipywidgets import Video, Image


cursor = evadb.connect().cursor()
warnings.filterwarnings("ignore")

cursor.query("DROP TABLE IF EXISTS HAPPY;").df()
cursor.query("LOAD VIDEO 'c001.mp4' INTO HAPPY").df()
cursor.query("""
    CREATE FUNCTION IF NOT EXISTS FakeTracker
    INPUT (frame NDARRAY UINT8(3, ANYDIM, ANYDIM)) 
    OUTPUT (labels NDARRAY STR(ANYDIM), scores NDARRAY FLOAT32(ANYDIM)) 
    TYPE FakeTrack IMPL 'fake_tracker.py';
""").df()

print(cursor.query("SELECT id, FakeTracker(data) FROM HAPPY WHERE id < 10").df())

cursor.query("DROP FUNCTION FakeTracker").df()