FROM vllm/vllm-openai:v0.20.2

COPY builder/requirements.txt /requirements.txt
RUN pip install --upgrade -r /requirements.txt

COPY src /src

WORKDIR /src

ENTRYPOINT []
CMD ["python3", "handler.py"]
