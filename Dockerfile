FROM eywalker/tensorflow-jupyter:v0.11.0rc0

RUN pip3 install seaborn
RUN pip3 install sklearn
RUN pip3 install scipy

WORKDIR /src

ADD . /src/learning_plasticity
RUN pip3 install -e /src/learning_plasticity
