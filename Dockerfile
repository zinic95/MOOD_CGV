FROM mood_test:latest

COPY ./*.py /workspace/
COPY ./shell/run_pixel* /workspace/
COPY ./shell/run_sample* /workspace/
COPY ./weights/* /workspace/weights/
RUN chmod +x /workspace/*.sh
