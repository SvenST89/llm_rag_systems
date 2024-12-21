#FROM continuumio/miniconda3
FROM mcr.microsoft.com/devcontainers/base:jammy
#FROM continuumio/miniconda3

WORKDIR /20241123_RAG_System

# Create the environment:
COPY requirements.txt .
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip poppler-utils tesseract-ocr

# For using unstructured to extract PDF content with OCR
#RUN apt-get install poppler-utils
#RUN apt-get install tesseract-ocr

RUN pip install --no-cache-dir -r requirements.txt