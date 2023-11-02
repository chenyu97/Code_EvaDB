file_path1 = 'c026_single_video_2'
file_path2 = 'c026_single_video_3'

# 打开文件进行比较
with open(file_path1, 'r') as file1, open(file_path2, 'r') as file2:
    # 逐行比较文件内容
    for line1, line2 in zip(file1, file2):
        if line1 != line2:
            print(f"Files are different.")
            break
    else:
        # 如果循环正常结束，说明至少两个文件有相同数量的行，并且每行都相同
        # 还需要检查是否有多余的行
        if file1.readline() == '' and file2.readline() == '':
            print(f"Files are identical.")
        else:
            print(f"Files are different.")
