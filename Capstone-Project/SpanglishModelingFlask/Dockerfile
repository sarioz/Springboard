FROM continuumio/miniconda:latest

WORKDIR /home/docker_conda_template

COPY environment.yml ./
COPY app.py ./
COPY boot.sh ./
COPY pos_tagger ./pos_tagger/
COPY sentiment_analyzer ./sentiment_analyzer/
COPY data ./data/
COPY static ./static/
COPY templates ./templates/

RUN chmod +x boot.sh

RUN conda env create -f environment.yml

RUN echo "source activate min_webapp" > ~/.bashrc
ENV PATH /opt/conda/envs/min_webapp/bin:$PATH

EXPOSE 5000

ENTRYPOINT ["./boot.sh"]
