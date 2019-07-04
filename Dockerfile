FROM openjdk:8-jdk-slim AS builder
# install required packages for DIANNE
RUN apt-get update && apt-get install -y --no-install-recommends \ 
	build-essential \
	cmake \
	gfortran \
	lib32z1-dev \
	&& rm -rf /var/lib/apt/lists/*
WORKDIR /usr/src/app
COPY . ./
RUN ./gradlew -x testOSGi build 
RUN ./gradlew export.all export.coordinator export.runtime.evaluator export.runtime.learner export.runtime.agent

FROM openjdk:8-jre-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
	libgfortran3 \
	libgomp1 \
	&& rm -rf /var/lib/apt/lists/*
WORKDIR /root/
COPY --from=builder /usr/src/app/tools/generated/distributions/executable/ ./
EXPOSE 8080
CMD ["java", "-jar", "all.jar"]
