@REM python -m venv .venv-yolov10
@REM .\.venv-yolov10\Scripts\activate.bat
call pip uninstall ultralytics roboflow -y
call pip install huggingface-hub
call pip install autopep8
call git clone https://github.com/THU-MIG/yolov10.git
call cd yolov10 && pip install -e . && cd ..
