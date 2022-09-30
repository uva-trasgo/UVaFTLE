FROM ubuntu:22.04
LABEL name="GenerateFTLData"
LABEL org.opencontainers.image.description="Generation of the data required to compute the FTL"
LABEL version="v1"
LABEL maintainer="serlohu@upv.es"

ENV TZ="Europe/Madrid"
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && \
    apt-get install -y  libx11-6 libgl1-mesa-glx libxrender1 python3.10-full python3-pip && \
    apt autoclean -y && \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/*


RUN pip3 install pyvista meshio scipy multiprocess joblib numpy argparse

# Copy sources to create
COPY mesh-generation.py /usr/bin/mesh-generation.py

ENTRYPOINT [ "/usr/bin/mesh-generation.py" ] 
