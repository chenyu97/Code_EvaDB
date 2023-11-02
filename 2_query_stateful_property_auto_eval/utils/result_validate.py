import json


if __name__ == '__main__':

    file_path_1 = './resources/offline_profile_1.json'
    data_1 = None
    with open(file_path_1, 'r') as file:
        data_1 = json.load(file)

    file_path_2 = './resources/offline_profile_2.json'
    data_2 = None
    with open(file_path_2, 'r') as file:
        data_2 = json.load(file)

    file_path_3 = './resources/offline_profile_3.json'
    data_3 = None
    with open(file_path_3, 'r') as file:
        data_3 = json.load(file)
    
    for key, value in data_1.items():
        is_identical_1 = (data_1[key]['color_result'] == data_2[key]['color_result']) & \
              (data_1[key]['type_result'] == data_2[key]['type_result']) & \
                (data_1[key]['direction_result'] == data_2[key]['direction_result'])
        is_identical_2 = (data_2[key]['color_result'] == data_3[key]['color_result']) & \
              (data_2[key]['type_result'] == data_3[key]['type_result']) & \
                (data_2[key]['direction_result'] == data_3[key]['direction_result'])
        if not (is_identical_1 & is_identical_2):
            print('different!')
            break
    
    print('finish!')
        