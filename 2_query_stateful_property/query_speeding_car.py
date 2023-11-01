import evadb
import warnings
import time


timestamp = time.time()
time_struct = time.localtime(timestamp)  # 将时间戳转换为 struct_time 对象
readable_time = time.strftime("%Y-%m-%d %H:%M:%S", time_struct)  # 将 struct_time 对象格式化为字符串

with open('./result/time_cost_D1_3min_' + readable_time, 'a') as file:

    def drop_defined(cursor):
        time_start = time.time()
        # ======
        cursor.query("DROP TABLE IF EXISTS MyVideo;").df()
        # ======
        time_end = time.time()
        file.write("drop_video: " + str(time_end - time_start) + '\n')


        time_start = time.time()
        # ======
        cursor.query("DROP TABLE IF EXISTS TrackResult;").df()
        # ======
        time_end = time.time()
        file.write("drop_table_TrackResult: " + str(time_end - time_start) + '\n')


        time_start = time.time()
        # ======
        cursor.query("DROP TABLE IF EXISTS TrackResultAdd1;").df()
        # ======
        time_end = time.time()
        file.write("drop_table_TrackResultAdd1: " + str(time_end - time_start) + '\n')


        time_start = time.time()
        # ======
        cursor.query("DROP TABLE IF EXISTS TrackResultJoin;").df()
        # ======
        time_end = time.time()
        file.write("drop_table_TrackResultJoin: " + str(time_end - time_start) + '\n')


        time_start = time.time()
        # ======
        cursor.query("DROP FUNCTION IF EXISTS Add1;").df()
        # ======
        time_end = time.time()
        file.write("drop_function_Add1: " + str(time_end - time_start) + '\n')


        time_start = time.time()
        # ======
        cursor.query("DROP FUNCTION IF EXISTS Velocity;").df()
        # ======
        time_end = time.time()
        file.write("drop_function_Velocity: " + str(time_end - time_start) + '\n')


        time_start = time.time()
        # ======
        cursor.query("DROP FUNCTION IF EXISTS Yolo;").df()
        # ======
        time_end = time.time()
        file.write("drop_function_Yolo: " + str(time_end - time_start) + '\n')


        time_start = time.time()
        # ======
        cursor.query("DROP FUNCTION IF EXISTS NorFairTracker;").df()
        # ======
        time_end = time.time()
        file.write("drop_function_NorFairTracker: " + str(time_end - time_start) + '\n')


    warnings.filterwarnings("ignore")


    time_total_start = time.time()


    time_start = time.time()
    # ======
    cursor = evadb.connect().cursor()
    # ======
    time_end = time.time()
    file.write("evadb_connect: " + str(time_end - time_start) + '\n')


    drop_defined(cursor)


    time_start = time.time()
    # ======
    load_video_query = """
                LOAD VIDEO 
                    './ua_detrac.mp4'
                INTO 
                    MyVideo;
    """
    cursor.query(load_video_query).df()
    # ======
    time_end = time.time()
    file.write("load_video: " + str(time_end - time_start) + '\n')


    time_start = time.time()
    # ======
    cursor.query("""
                CREATE FUNCTION 
                    Add1
                IMPL 
                    './add1.py';
    """).df()
    # ======
    time_end = time.time()
    file.write("create_function_add1: " + str(time_end - time_start) + '\n')


    time_start = time.time()
    # ======
    cursor.query("""
                CREATE FUNCTION 
                    Velocity
                IMPL
                    './velocity.py';
            """).df()
    # ======
    time_end = time.time()
    file.write("create_function_velocity: " + str(time_end - time_start) + '\n')


    time_start = time.time()
    # ======
    cursor.query("""
        CREATE FUNCTION Yolo
        IMPL './yolo_detector.py'
    """).df()
    # ======
    time_end = time.time()
    file.write("create_function_Yolo: " + str(time_end - time_start) + '\n')


    time_start = time.time()
    # ======
    cursor.query("""
        CREATE FUNCTION NorFairTracker
        IMPL './nor_fair.py'
    """).df()
    # ======
    time_end = time.time()
    file.write("create_function_NorFairTracker: " + str(time_end - time_start) + '\n')


    time_start = time.time()
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
    time_end = time.time()
    file.write("create_table_TrackResult: " + str(time_end - time_start) + '\n')


    time_start = time.time()
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
    time_end = time.time()
    file.write("create_table_TrackResultAdd1: " + str(time_end - time_start) + '\n')


    time_start = time.time()
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
    time_end = time.time()
    file.write("Join_table: " + str(time_end - time_start) + '\n')


    time_start = time.time()
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
    time_end = time.time()
    file.write("query: " + str(time_end - time_start) + '\n')


    time_start = time.time()
    # ======
    response.to_csv('./result/speeding_car_D1_3min.csv', index=False)
    # ======
    time_end = time.time()
    file.write("save: " + str(time_end - time_start))


    drop_defined(cursor)


    time_total_end = time.time()
    file.write("total time: " + str(time_total_end - time_total_start) + '\n')