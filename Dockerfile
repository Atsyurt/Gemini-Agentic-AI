FROM python:3.11
#docker build --tag agentic-ai-gemini-fastapi .
WORKDIR /code

COPY requirements.txt .

RUN pip install langchain==0.2.16 langchain-chroma==0.1.3 langchain-community==0.2.16 langchain-core==0.2.38 langchain-huggingface==0.0.3 langchain-text-splitters==0.2.4 numpy==1.26.4 spacy==3.8.0
RUN pip install langchain-google-community langchain-google-genai langgraph fastapi uvicorn jinja2 python-multipart



COPY . .

EXPOSE 80

CMD ["uvicorn", "main:app", "--reload","--host", "0.0.0.0","--port", "80"]