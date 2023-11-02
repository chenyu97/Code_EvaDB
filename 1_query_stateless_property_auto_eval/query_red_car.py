import evadb
import time
import warnings

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", help="path to video file")
parser.add_argument("--save_folder", help="path to save query result")
args = parser.parse_args()

with open(f"{args.save_folder}/time_{time.strftime('%Y.%m.%d.%H.%M.%S')}.txt", 'w+') as file:

    def drop_defined(cursor):
        time_start_0 = time.time()
        # ======
        cursor.query("DROP TABLE IF EXISTS MyVideo;").df()
        # ======
        time_end_0 = time.time()
        file.write("drop_video: " + str(time_end_0 - time_start_0) + '\n')

        time_start_1 = time.time()
        # ======
        cursor.query("DROP FUNCTION IF EXISTS Color;").df()
        # ======
        time_end_1 = time.time()
        file.write("drop_function_Color: " +
                   str(time_end_1 - time_start_1) + '\n')

        time_start_2 = time.time()
        # ======
        cursor.query("DROP FUNCTION IF EXISTS Yolo;").df()
        # ======
        time_end_2 = time.time()
        file.write("drop_function_Yolo: " +
                   str(time_end_2 - time_start_2) + '\n')

        time_start_3 = time.time()
        # ======
        cursor.query("DROP FUNCTION IF EXISTS NorFairTracker;").df()
        # ======
        time_end_3 = time.time()
        file.write("drop_function_NorFairTracker: " +
                   str(time_end_3 - time_start_3) + '\n')

    warnings.filterwarnings("ignore")

    time_total_start = time.time()

    time_start = time.time()
    # ======
    cursor = evadb.connect().cursor()
    # ======
    time_end = time.time()
    file.write("evadb_connect: " + str(time_end - time_start) + '\n')

    drop_defined(cursor)

    time_start_10 = time.time()
    # ======
    load_video_query = f"""
                    LOAD VIDEO 
                        '{args.path}'
                    INTO 
                        MyVideo;
                    """
    cursor.query(load_video_query).df()
    # ======
    time_end_10 = time.time()
    file.write("load_video: " + str(time_end_10 - time_start_10) + '\n')

    time_start_11 = time.time()
    # ======
    cursor.query("""
        CREATE FUNCTION Color
        IMPL './color_NN.py'
    """).df()
    # ======
    time_end_11 = time.time()
    file.write("create_function_color: " +
               str(time_end_11 - time_start_11) + '\n')

    time_start_12 = time.time()
    # ======
    cursor.query("""
        CREATE FUNCTION Yolo
        IMPL './yolo_detector.py'
    """).df()
    # ======
    time_end_12 = time.time()
    file.write("create_function_Yolo: " +
               str(time_end_12 - time_start_12) + '\n')

    time_start_13 = time.time()
    # ======
    cursor.query("""
        CREATE FUNCTION NorFairTracker
        IMPL './nor_fair.py'
    """).df()
    # ======
    time_end_13 = time.time()
    file.write("create_function_NorFairTracker: " +
               str(time_end_13 - time_start_13) + '\n')

    time_start_14 = time.time()
    # ======
    select_query = """
                SELECT 
                    id, T.bbox
                FROM 
                    MyVideo 
                JOIN LATERAL
                    UNNEST(EXTRACT_OBJECT(data, Yolo, NorFairTracker)) 
                AS 
                    T(iid, label, bbox, score)
                WHERE
                    Color(Crop(data, T.bbox)) = 'red'
                AND
                    T.label = 'car'
                """
    response = cursor.query(select_query).df()
    # ======
    time_end_14 = time.time()
    file.write("query: " + str(time_end_14 - time_start_14) + '\n')

    time_start_15 = time.time()
    # ======
    response.to_csv(
        f"{args.save_folder}/result_{time.strftime('%Y.%m.%d.%H.%M.%S')}.csv", index=False)
    # ======
    time_end_15 = time.time()
    file.write("save: " + str(time_end_15 - time_start_15) + '\n')

    drop_defined(cursor)

    time_total_end = time.time()
    file.write("total time: " + str(time_total_end - time_total_start) + '\n')

    file.write(f"from load to save time: {time_end_15 - time_start_10}\n")
    print(f"from load to save time: {time_end_15 - time_start_10}\n")
