FROM python:3.9.18-slim

WORKDIR /code

COPY ./requirements_without_jupyter.txt /code/requirements_without_jupyter.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements_without_jupyter.txt

COPY ./src /code/src

CMD ["python", "app/main.py"]