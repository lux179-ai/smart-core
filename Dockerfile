# Usa un'immagine Python leggera e sicura
FROM python:3.11-slim

# Imposta la directory di lavoro
WORKDIR /app

# Variabili d'ambiente per evitare file .pyc e output bufferizzato
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copia i requirements e installa le dipendenze
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia il codice sorgente
COPY main.py .

# Espone la porta standard di FastAPI
EXPOSE 8000

# Comando di avvio del server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
