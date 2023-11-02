import json
from collections import deque, Counter


def not_color(data, color):
    return_keys = []
    for key, value in data.items():
        if value['color_result'] != color:
            return_keys.append(key)
    return return_keys

def not_type(data, type):
    return_keys = []
    for key, value in data.items():
        if value['type_result'] != type:
            return_keys.append(key)
    return return_keys

def not_direction(data, direction):
    return_keys = []
    for key, value in data.items():
        if value['direction_result'] != direction:
            return_keys.append(key)
    return return_keys

def color_and_not_type(data, color, type):
    return_keys = []
    for key, value in data.items():
        if (value['color_result'] == color) & (value['type_result'] != type):
            return_keys.append(key)
    return return_keys

def color_and_not_direction(data, color, direction):
    return_keys = []
    for key, value in data.items():
        if (value['color_result'] == color) & (value['direction_result'] != direction):
            return_keys.append(key)
    return return_keys

def type_and_not_color(data, type, color):
    return_keys = []
    for key, value in data.items():
        if (value['type_result'] == type) & (value['color_result'] != color):
            return_keys.append(key)
    return return_keys

def type_and_not_direction(data, type, direction):
    return_keys = []
    for key, value in data.items():
        if (value['type_result'] == type) & (value['direction_result'] != direction):
            return_keys.append(key)
    return return_keys

def direction_and_not_color(data, direction, color):
    return_keys = []
    for key, value in data.items():
        if (value['direction_result'] == direction) & (value['color_result'] != color):
            return_keys.append(key)
    return return_keys

def direction_and_not_type(data, direction, type):
    return_keys = []
    for key, value in data.items():
        if (value['direction_result'] == direction) & (value['type_result'] != type):
            return_keys.append(key)
    return return_keys

def optimal_order(data, query_color, query_type, query_direction):    
    color_order_opt, type_order_opt, direction_order_opt = -1, -1, -1
    save_time_max = 0
    order_option = [0, 1, 2]
    for color_order in order_option:
        for type_order in order_option:
            if type_order == color_order:
                continue
            for direction_order in order_option:
                if direction_order == color_order or direction_order == type_order:
                    continue
                save_time = 0
                if color_order == 0:
                    if type_order == 1:
                        return_keys = not_color(data, query_color)
                        for k in return_keys:
                            save_time += data[k]['type_cost'] + data[k]['direction_cost']
                        return_keys = color_and_not_type(data, query_color, query_type)
                        for k in return_keys:
                            save_time += data[k]['direction_cost']
                    elif direction_order == 1:
                        return_keys = not_color(data, query_color)
                        for k in return_keys:
                            save_time += data[k]['direction_cost'] + data[k]['type_cost']
                        return_keys = color_and_not_direction(data, query_color, query_direction)
                        for k in return_keys:
                            save_time += data[k]['type_cost']
                elif type_order == 0:
                    if color_order == 1:
                        return_keys = not_type(data, query_type)
                        for k in return_keys:
                            save_time += data[k]['color_cost'] + data[k]['direction_cost']
                        return_keys = type_and_not_color(data, query_type, query_color)
                        for k in return_keys:
                            save_time += data[k]['direction_cost']
                    elif direction_order == 1:
                        return_keys = not_type(data, query_type)
                        for k in return_keys:
                            save_time += data[k]['direction_cost'] + data[k]['color_cost']
                        return_keys = type_and_not_direction(data, query_type, query_direction)
                        for k in return_keys:
                            save_time += data[k]['color_cost']
                elif direction_order == 0:
                    if color_order == 1:
                        return_keys = not_direction(data, query_direction)
                        for k in return_keys:
                            save_time += data[k]['color_cost'] + data[k]['type_cost']
                        return_keys = direction_and_not_color(data, query_direction, query_color)
                        for k in return_keys:
                            save_time += data[k]['type_cost']
                    elif type_order == 1:
                        return_keys = not_direction(data, query_direction)
                        for k in return_keys:
                            save_time += data[k]['type_cost'] + data[k]['color_cost']
                        return_keys = direction_and_not_type(data, query_direction, query_type)
                        for k in return_keys:
                            save_time += data[k]['color_cost']
                if (color_order == 1) & (type_order == 2) & (direction_order == 0):
                    print('120: ' + str(save_time))
              
                if save_time > save_time_max:
                    save_time_max = save_time
                    color_order_opt, type_order_opt, direction_order_opt = color_order, type_order, direction_order
    
    return color_order_opt, type_order_opt, direction_order_opt


if __name__ == '__main__':

    file_path = './resources/offline_profile_1.json'
    data = None
    with open(file_path, 'r') as file:
        data = json.load(file)

    query_color, query_type, query_direction = 'black', 'suv', 'straight'

    order = optimal_order(data, query_color, query_type, query_direction)
    print(str(order))