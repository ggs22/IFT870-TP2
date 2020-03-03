#!/bin/bash

py_file_name='tp2.py'

if [ -e  $PWD/$py_file_name ]
then
	date_str=$(date +"%Y_%m_%d_%H:%M:%S")
	cp $PWD/$py_file_name $PWD/'tp2_backup_'$date_str'.py'
fi

ipynb-py-convert ./tp2.ipynb ./$py_file_name
