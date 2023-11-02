import evadb
import warnings
import time

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
        cursor.query("DROP TABLE IF EXISTS TrackResult;").df()
        # ======
        time_end_1 = time.time()
        file.write("drop_table_TrackResult: " +
                   str(time_end_1 - time_start_1) + '\n')

        time_start_2 = time.time()
        # ======
        cursor.query("DROP TABLE IF EXISTS TrackResultAdd1;").df()
        # ======
        time_end_2 = time.time()
        file.write("drop_table_TrackResultAdd1: " +
                   str(time_end_2 - time_start_2) + '\n')

        time_start_3 = time.time()
        # ======
        cursor.query("DROP TABLE IF EXISTS TrackResultJoin;").df()
        # ======
        time_end_3 = time.time()
        file.write("drop_table_TrackResultJoin: " +
                   str(time_end_3 - time_start_3) + '\n')

        time_start_4 = time.time()
        # ======
        cursor.query("DROP FUNCTION IF EXISTS Add1;").df()
        # ======
        time_end_4 = time.time()
        file.write("drop_function_Add1: " +
                   str(time_end_4 - time_start_4) + '\n')

        time_start_5 = time.time()
        # ======
        cursor.query("DROP FUNCTION IF EXISTS Velocity;").df()
        # ======
        time_end_5 = time.time()
        file.write("drop_function_Velocity: " +
                   str(time_end_5 - time_start_5) + '\n')

        time_start_6 = time.time()
        # ======
        cursor.query("DROP FUNCTION IF EXISTS Yolo;").df()
        # ======
        time_end_6 = time.time()
        file.write("drop_function_Yolo: " +
                   str(time_end_6 - time_start_6) + '\n')

        time_start_7 = time.time()
        # ======
        cursor.query("DROP FUNCTION IF EXISTS NorFairTracker;").df()
        # ======
        time_end_7 = time.time()
        file.write("drop_function_NorFairTracker: " +
                   str(time_end_7 - time_start_7) + '\n')

    warnings.filterwarnings("ignore")

    time_total_start = time.time()

    time_start_10 = time.time()
    # ======
    cursor = evadb.connect().cursor()
    # ======
    time_end_10 = time.time()
    file.write("evadb_connect: " + str(time_end_10 - time_start_10) + '\n')

    drop_defined(cursor)

    time_start_11 = time.time()
    # ======
    load_video_query = f"""
                LOAD VIDEO 
                    '{args.path}'
                INTO 
                    MyVideo;
    """
    cursor.query(load_video_query).df()
    # ======
    time_end_11 = time.time()
    file.write("load_video: " + str(time_end_11 - time_start_11) + '\n')

    time_start_12 = time.time()
    # ======
    cursor.query("""
                CREATE FUNCTION 
                    Add1
                IMPL 
                    './add1.py';
    """).df()
    # ======
    time_end_12 = time.time()
    file.write("create_function_add1: " +
               str(time_end_12 - time_start_12) + '\n')

    time_start_13 = time.time()
    # ======
    cursor.query("""
                CREATE FUNCTION 
                    Velocity
                IMPL
                    './velocity.py';
            """).df()
    # ======
    time_end_13 = time.time()
    file.write("create_function_velocity: " +
               str(time_end_13 - time_start_13) + '\n')

    time_start_14 = time.time()
    # ======
    cursor.query("""
        CREATE FUNCTION Yolo
        IMPL './yolo_detector.py'
    """).df()
    # ======
    time_end_14 = time.time()
    file.write("create_function_Yolo: " +
               str(time_end_14 - time_start_14) + '\n')

    time_start_15 = time.time()
    # ======
    cursor.query("""
        CREATE FUNCTION NorFairTracker
        IMPL './nor_fair.py'
    """).df()
    # ======
    time_end_15 = time.time()
    file.write("create_function_NorFairTracker: " +
               str(time_end_15 - time_start_15) + '\n')

    time_start_16 = time.time()
    # ======
    cursor.query("""
            CREATE TABLE 
                TrackResult
            AS 
                SELECT 
                    id, T.iid, T.bbox, T.score, T.label
                FROM 
                    MyVideo 
                JOIN LATERAL 
                    UNNEST(
                        EXTRACT_OBJECT(
                            data, Yolo, NorFairTracker
                        )
                    ) 
                AS 
                    T(iid, label, bbox, score)
    """).df()
    # ======
    time_end_16 = time.time()
    file.write("create_table_TrackResult: " +
               str(time_end_16 - time_start_16) + '\n')

    time_start_17 = time.time()
    # ======
    cursor.query("""
                CREATE TABLE 
                    TrackResultAdd1
                AS 
                    SELECT 
                        Add1(id, iid, bbox)
                    FROM 
                        TrackResult
            """).df()
    # ======
    time_end_17 = time.time()
    file.write("create_table_TrackResultAdd1: " +
               str(time_end_17 - time_start_17) + '\n')

    time_start_18 = time.time()
    # ======
    join = """
            CREATE TABLE 
                TrackResultJoin
            AS 
                SELECT 
                    trackresult.id,
                    trackresult.iid,
                    trackresult.bbox,
                    trackresult.label,
                    trackresult.score,
                    trackresultadd1.last_bbox
                FROM
                    TrackResult
                JOIN 
                    TrackResultAdd1 
                ON 
                    trackresult.id = trackresultadd1.added_id 
                AND 
                    trackresult.iid = trackresultadd1.cur_iid
            """
    response = cursor.query(join).df()
    # ======
    time_end_18 = time.time()
    file.write("Join_table: " + str(time_end_18 - time_start_18) + '\n')

    time_start_19 = time.time()
    # ======
    response = cursor.query("""
            SELECT 
                id, iid, bbox
            FROM
                TrackResultJoin
            WHERE
                Velocity(bbox, last_bbox) > 1
            AND
                label = 'car'
    """).df()
    # ======
    time_end_19 = time.time()
    file.write("query: " + str(time_end_19 - time_start_19) + '\n')

    time_start_20 = time.time()
    # ======
    response.to_csv(
        f"{args.save_folder}/speeding_car_result_{time.strftime('%Y.%m.%d.%H.%M.%S')}.csv", index=False
    )
    # ======
    time_end_20 = time.time()
    file.write("save: " + str(time_end_20 - time_start_20))

    drop_defined(cursor)

    time_total_end = time.time()
    file.write("total time: " + str(time_total_end - time_total_start) + '\n')

    file.write(f"from load to save time: {time_end_20 - time_start_11}\n")
    print(f"from load to save time: {time_end_20 - time_start_11}\n")
