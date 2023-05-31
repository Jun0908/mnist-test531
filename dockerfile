# Dockerfile
FROM python:3.9

# apt
RUN apt update
RUN apt install -y bzip2 wget git

WORKDIR /app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "mnist.py"]