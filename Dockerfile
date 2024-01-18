FROM ubuntu:20.04
RUN apt-get update && apt-get install -y git sudo tmux
ENV TZ=Europe/Stockholm
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt install -y libgl1-mesa-dev libgl1-mesa-glx libglew-dev libosmesa6-dev software-properties-common net-tools xpra xserver-xorg-dev libglfw3-dev patchelf wget python3-pip

RUN git clone https://github.com/jstyrud/BeBOP
WORKDIR /BeBOP
COPY ./scripts/install.sh install.sh
RUN /bin/bash -c "./install.sh"

ENTRYPOINT ["/bin/bash"]