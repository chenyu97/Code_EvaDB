import os

def files_in_directory(dir_path):
    """返回目录中所有文件的列表，按文件名排序"""
    return sorted([f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))])

def files_are_identical(file1_path, file2_path):
    """比较两个文件的内容是否完全相同"""
    with open(file1_path, 'rb') as f1, open(file2_path, 'rb') as f2:
        return f1.read() == f2.read()

dir_a_path = './results'
dir_b_path = './results_vqpy_old'

files_in_a = files_in_directory(dir_a_path)

count = 0
modified_files_in_a = []
for file in files_in_a:
    if count % 5 in {0,1,2,3,4}:
        modified_files_in_a.append(file)
    count += 1
files_in_a = modified_files_in_a

files_in_b = files_in_directory(dir_b_path)

if len(files_in_a) != len(files_in_b):
    print("文件数量不同，无法进行比较！")
else:
    for file_a, file_b in zip(files_in_a, files_in_b):
        file_a_path = os.path.join(dir_a_path, file_a)
        file_b_path = os.path.join(dir_b_path, file_b)
        
        if files_are_identical(file_a_path, file_b_path):
            print(f"文件 {file_a} 和 {file_b} 是相同的")
        else:
            print(f"文件 {file_a} 和 {file_b} 是不同的")

