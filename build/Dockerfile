FROM python:3.10-bookworm

RUN apt update 
RUN apt install -y ca-certificates curl
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"
RUN poetry config virtualenvs.in-project true