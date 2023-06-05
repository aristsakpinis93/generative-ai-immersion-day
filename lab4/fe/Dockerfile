#FROM --platform=linux/x86-64 python:3.11.1-slim
FROM python:3.11.1-slim

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY app.py app.py

COPY aws.png aws.png

EXPOSE 80

ENTRYPOINT [ "streamlit", "run", "app.py", \
    "--server.enableCORS", "true", \
             "--server.port", "80", \
             "--browser.serverPort", "80"]