# Use a specific and stable Python version
FROM python:3.11-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy dependencies first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what is needed at runtime
COPY src ./src
COPY model ./model
COPY templates ./templates
COPY app.py .

# Expose the port the app runs on
EXPOSE 8080

# Use gunicorn to run the app
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
