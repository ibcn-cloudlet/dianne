FROM ubuntu:14.04

# update and upgrade apt packages
RUN apt-get update
RUN apt-get -y upgrade

# install missing packages
RUN apt-get install -y python-software-properties software-properties-common git ant

# install java 8
RUN \
  echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | debconf-set-selections && \
  add-apt-repository -y ppa:webupd8team/java && \
  apt-get update && \
  apt-get install -y oracle-java8-installer && \
  rm -rf /var/lib/apt/lists/* && \
  rm -rf /var/cache/oracle-jdk8-installer

# create dianne user + home dir
RUN /usr/sbin/useradd --create-home --home-dir /home/dianne --shell /bin/false dianne
ENV home /home/dianne
WORKDIR ${home}

# copy all dianne code
ADD ./ $home

# build dianne runtime
RUN ant clean build

# run
ENTRYPOINT ["ant"]
