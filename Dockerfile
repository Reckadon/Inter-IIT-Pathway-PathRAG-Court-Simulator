FROM pathwaycom/pathway:latest

WORKDIR /app

COPY requirements.txt ./
RUN pip install -U --no-cache-dir -r requirements.txt

RUN pip install -U --no-cache-dir --upgrade pathway

COPY . .

EXPOSE 8766

CMD ["python", "./main.py"]
