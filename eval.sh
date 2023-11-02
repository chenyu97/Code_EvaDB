#!/bin/bash

# assume we're at /home/chenyu97/Codes/eval.sh
working_dir=/home/chenyu97/Codes
cd $working_dir

# prepare conda
eval "$(conda shell.bash hook)"

# basic directories
dataset_dir=/home/chenyu97/Codes/Three_Datasets
top_result_dir=$working_dir/eval_results

dataset_1_dir=$dataset_dir/Banff
dataset_1_cutted_dir=$dataset_1_dir/cut_videos
dataset_1_video_list=($dataset_1_cutted_dir/*)

dataset_2_dir=$dataset_dir/Jackson
dataset_2_cutted_dir=$dataset_2_dir/cut_videos
dataset_2_video_list=($dataset_2_cutted_dir/*.mp4)

dataset_3_dir=$dataset_dir/Southampton
dataset_3_cutted_dir=$dataset_3_dir/cut_videos
dataset_3_video_list=($dataset_3_cutted_dir/*.mp4)

# define vqpy dirs
vqpy_working_dir=/home/chenyu97/Codes/vqpy/examples/compared_sql_query_auto_eval

vqpy_q1_query=$vqpy_working_dir/Q1_Query_Red_Car.py
vqpy_q2_query=$vqpy_working_dir/Q2_Query_Speeding_Car.py
vqpy_q3_query=$vqpy_working_dir/Q3_Query_red_speeding_car.py

# define evadb dirs
evadb_dir=/home/chenyu97/Codes/Code_EvaDB

evadb_q1_dir=$evadb_dir/1_query_stateless_property_auto_eval
evadb_q1_query=$evadb_q1_dir/query_red_car.py

evadb_q2_dir=$evadb_dir/2_query_stateful_property_auto_eval
evadb_q2_query=$evadb_q2_dir/query_speeding_car.py

evadb_q3_dir=$evadb_dir/3_query_stateless_and_stateful_properties_auto_eval
evadb_q3_query=$evadb_q3_dir/query_red_speeding_car.py

########################### EDIT ####################################
# choose dataset and query
dataset=1
query=1
video_list=("${dataset_1_video_list[@]}")
vqpy_query=$vqpy_q1_query
evadb_dir=$evadb_q1_dir
evadb_query=$evadb_q1_query
result_dir=$top_result_dir/dataset_1_query_1
########################### EDIT END ####################################

# make all dirs
mkdir -p $top_result_dir
mkdir -p $result_dir
for video in "${video_list[@]}"; do
	video_name=$(basename "$video")
	cur_result_dir=$result_dir/$video_name
	cur_result_dir_vqpy=$cur_result_dir/vqpy
	cur_result_dir_evadb=$cur_result_dir/evadb
	mkdir -p $cur_result_dir
	mkdir -p $cur_result_dir_vqpy
	mkdir -p $cur_result_dir_evadb
done

# activate evadb_venv conda env and switch to evadb working dir
conda activate evadb_venv
cd $evadb_dir

# run evadb query with variables passed in:
# --path: path to video file
# --save_time_file: path to save time
for video in "${video_list[@]}"; do
	video_name=$(basename "$video")
	echo "running evadb query on $video_name"
	cur_result_dir=$result_dir/$video_name/evadb
	python $evadb_query --path $video --save_folder $cur_result_dir
	# clean up evadb_data and hope the disk space is enough
	rm -rf $evadb_dir/evadb_data
done

# # activate vqpy conda and switch working dir
conda activate vqpy
cd $vqpy_working_dir

# run vqpy query with variables passed in:
# --path: path to video file
# --save_folder: path to save query result
# --save_time_file: path to save time
for video in "${video_list[@]}"; do
	video_name=$(basename "$video")
	echo "running vqpy query on $video_name"
	cur_result_dir=$result_dir/$video_name/vqpy
	python $vqpy_query --path $video --save_folder $cur_result_dir
done


