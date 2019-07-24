#!/bin/sh
current_dir=`pwd`
export PYTHONPATH="${current_dir}/../../build/release/bindings/" 
python main.py ~/Downloads
