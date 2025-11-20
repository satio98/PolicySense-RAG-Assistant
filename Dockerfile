FROM python:3.10-slim-buster

WORKDIR /legal_app

COPY . /legal_app

RUN pip install -r requirements.txt

CMD ["python3", "legal_app.py"]