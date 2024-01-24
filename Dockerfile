#So this dockerfile does containerize training. So whenever you are running your docker container, you can actually make your
#docker container use your GPU to train your model. Though we will not be doing that with this project, just know this is how you do it
FROM nvidia/cuda:11.3.0-cudnn8-runtime-ubuntu20.04

RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y curl
RUN apt-get install unzip
RUN apt-get -y install python3
RUN apt-get -y install python3-pip

WORKDIR ./engine

COPY . .

RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["python3","src/pipeline/pipeline.py"]