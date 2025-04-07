FROM tensorflow/tensorflow:latest-gpu
RUN pip3 install keras==3.6.0
RUN pip3 install numpy==1.24.3
RUN pip3 install pandas==2.0.3