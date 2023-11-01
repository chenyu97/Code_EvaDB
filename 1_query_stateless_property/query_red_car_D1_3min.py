import evadb
import time
import warnings


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
        cursor.query("DROP FUNCTION IF EXISTS Color;").df()
        # ======
        time_end = time.time()
        file.write("drop_function_Color: " + str(time_end - time_start) + '\n')


        time_start = time.time()
        # ======
        cursor.query("DROP FUNCTION IF EXISTS Yolo;").df()
        # ======
        time_end = time.time()
        file.write("drop_function_Yolo: " + str(time_end - time_start) + '\n')


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
                        '../../Three_Datasets/Banff/cut_videos/3min_banff_sat_am001.mp4'
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
        CREATE FUNCTION Color
        IMPL './color_NN.py'
    """).df()
    # ======
    time_end = time.time()
    file.write("create_function_color: " + str(time_end - time_start) + '\n')


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
    select_query = """
                SELECT 
                    id, D.bbox
                FROM 
                    MyVideo 
                JOIN LATERAL
                    UNNEST(Yolo(data)) 
                AS 
                    D(label, bbox, score)
                WHERE
                    Color(Crop(data, D.bbox)) = 'red'
                AND
                    D.label = 'car'
                """
    response = cursor.query(select_query).df()
    # ======
    time_end = time.time()
    file.write("query: " + str(time_end - time_start) + '\n')


    time_start = time.time()
    # ======
    response.to_csv('./result/red_car_D1_3min.csv', index=False)
    # ======
    time_end = time.time()
    file.write("save: " + str(time_end - time_start) + '\n')


    drop_defined(cursor)


    time_total_end = time.time()
    file.write("total time: " + str(time_total_end - time_total_start) + '\n')