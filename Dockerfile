# docker build -t bootcamp_fix:0.01 .
# docker run -it --rm -p 8888:8888 -v ${PWD}:/workdir bootcamp_fix:0.01
FROM ucbbar/chisel-bootcamp:latest

RUN cp -r /coursier_cache /home/bootcamp/coursier_cache
ENV COURSIER_CACHE=/home/bootcamp/coursier_cache
RUN pip install RISE

USER bootcamp
WORKDIR /workdir

EXPOSE 8888
CMD jupyter notebook --no-browser --ip 0.0.0.0 --port 8888
