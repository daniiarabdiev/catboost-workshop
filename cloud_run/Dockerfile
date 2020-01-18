FROM python:3.7

# One can name container in any way
ENV APP_HOME /catboostworkshop
WORKDIR $APP_HOME
# copies entire directory
COPY . ./

# Installs libraries in requirements.txt
RUN pip install -r requirements.txt

# makes sure the default service is used
ENV PATH $PATH:/root/tools/google-cloud-sdk/bin
RUN echo '[GoogleCompute]\nservice_account = default' > /etc/boto.cfg

CMD exec gunicorn --bind :8080 --workers 1 --threads 8 app:app