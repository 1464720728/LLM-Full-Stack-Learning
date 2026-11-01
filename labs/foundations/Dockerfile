FROM python:3.12-slim
WORKDIR /app
COPY labs/__init__.py /app/labs/__init__.py
COPY labs/foundations /app/labs/foundations
USER 10001:10001
ENTRYPOINT ["python", "-m", "labs.foundations.log_project"]
