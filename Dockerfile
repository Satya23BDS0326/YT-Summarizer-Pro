# Use an official Python runtime
FROM python:3.10-slim

# Install ffmpeg (required by yt-dlp to extract audio)
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your entire original project into the container
COPY . .

# Expose the port Hugging Face uses
EXPOSE 7860

# Run your original backend.py exactly as it is
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "7860"]