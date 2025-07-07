FROM python:3.11

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install "ultralytics<=8.3.40" 
RUN pip install opencv-python 
RUN pip install matplotlib 
RUN pip install gdown 
RUN pip install streamlit 
RUN pip install importnb
