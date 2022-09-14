FROM python:latest
WORKDIR /app
RUN apt update
RUN apt install -y openslide-tools
COPY requirements.txt requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt
COPY . .
CMD ["python","./ray_core.py"]
