FROM eywalker/tensorflow-jupyter:v0.11.0rc0

RUN pip3 install seaborn
RUN pip3 install sklearn
RUN pip3 install scipy
RUN pip3 install --upgrade tensorflow
RUN pip3 install datajoint

WORKDIR /src

ADD . /src/Plasticity_Ker/PlasticityKer
RUN pip3 install -e /src/Plasticity_Ker/PlasticityKer
