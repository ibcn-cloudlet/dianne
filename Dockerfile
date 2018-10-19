FROM ubuntu:14.04

# add dianne user
RUN /usr/sbin/useradd --create-home --home-dir /home/dianne --shell /bin/false dianne
ENV home /home/dianne
WORKDIR ${home}


# update and upgrade apt packages
RUN apt-get update && \ 
    apt-get -y upgrade

# install required packages for DIANNE
RUN apt-get install -y \ 
	build-essential \
	cmake \
	git \
	gfortran


# install Java 8
RUN \
  echo "deb http://ppa.launchpad.net/webupd8team/java/ubuntu trusty main" > /etc/apt/sources.list.d/webudp8team-java-trusty.list && \
  apt-key adv --keyserver keyserver.ubuntu.com --recv-keys C2518248EEA14886 && \
  echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | debconf-set-selections && \
  apt-get update && \
  apt-get install -y oracle-java8-installer && \
  rm -rf /var/lib/apt/lists/* && \
  rm -rf /var/cache/oracle-jdk8-installer && \
  ln -s /usr/lib/jvm/java-8-oracle/include/linux/jni_md.h /usr/lib/jvm/java-8-oracle/include/jni_md.h && \
  ln -s /usr/lib/jvm/java-8-oracle/include/linux/jawt_md.h /usr/lib/jvm/java-8-oracle/include/jawt_md.h
ENV JAVA_HOME /usr/lib/jvm/java-8-oracle


# add DIANNE code
ADD ./ $home
RUN chown -R dianne:dianne $home


# switch to dianne user
USER dianne


# build DIANNE, exclude ALE
RUN ./gradlew cleanAll -x :be.iminds.iot.dianne.rl.environment.ale:assemble assemble :tools:export.all


# set default bndrun target
ENV TARGET all

EXPOSE 8080

# run
ENTRYPOINT ["/home/dianne/entrypoint.sh"]
