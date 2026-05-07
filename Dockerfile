FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir "numpy<2.0.0"

COPY src/ ./src/
COPY configs/ ./configs/

# Strip local Windows absolute paths from config files so they work correctly inside the Linux container
RUN sed -i 's|C:/Users/sarva/.vscode/Styarth sir dataset/||g' configs/inference.yaml

RUN mkdir -p runs/classify/solar_defects/exp/weights

COPY runs/classify/solar_defects/exp/weights/best.pt runs/classify/solar_defects/exp/weights/best.pt

RUN mkdir -p logs

EXPOSE 8000

ENV PYTHONPATH=/app

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]