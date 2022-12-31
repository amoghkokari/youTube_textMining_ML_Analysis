FROM python:3.8

RUN pip install -r requirements.txt

WORKDIR /app

COPY . .

EXPOSE 5001

CMD python3 ./main.py