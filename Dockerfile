FROM python:3.11.0b3-bullseye


USER 0

# setup working directory for jenkins_api
RUN mkdir /insmodel
WORKDIR /insmodel/
ADD src /insmodel/
RUN chmod 777 -R '/insmodel/'

# Install MySQL Dependency
RUN yum install python-pip

# Install upgrade pip
RUN pip install --upgrade pip


# install dependencies
COPY requirements.txt ./
RUN pip install -r requirements.txt

EXPOSE 8080

CMD ["gunicorn", "--bind=0.0.0.0:8080", "--worker-class=gevent", "--worker-connections=500", "--workers=3" , "app:app"]
