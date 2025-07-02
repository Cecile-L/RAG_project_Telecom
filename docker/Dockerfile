# Python 3.9 officiel, compatible Mac M1
FROM python:3.9-slim

# Dépendances système 
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Dossier de travail
WORKDIR /app

# Copier les dépendances
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier les scripts
COPY . .
