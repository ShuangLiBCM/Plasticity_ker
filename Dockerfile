FROM eywalker/tensorflow-jupyter:v0.11.0rc0

RUN pip3 install seaborn
RUN pip3 install sklearn
RUN pip3 install scipy
RUN apt-get install -y git
RUN pip3 install git+https://github.com/datajoint/datajoint-python.git

WORKDIR /src
RUN git clone https://github.com/atlab/atflow.git &&\
    pip3 install /src/atflow

ADD . /src/Plasticity_Ker/PlasticityKer
RUN pip3 install -e /src/Plasticity_Ker/PlasticityKer
