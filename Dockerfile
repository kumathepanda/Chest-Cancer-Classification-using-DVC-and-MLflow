# Use a specific and stable Python version
FROM python:3.11-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker's layer caching
COPY requirements.txt .

# Install the Python dependencies
# --no-cache-dir reduces the image size
RUN pip install --no-cache-dir -r requirements.txt

# Now, copy the rest of the application code
# This includes your 'src', 'model', 'templates', and 'app.py' files
COPY ./src ./src
COPY ./model ./model
COPY ./templates ./templates
COPY ./app.py .

# Expose the port the app runs on
EXPOSE 8080

# Use gunicorn to run the app, which is a production-ready server
# This command is taken from your startup.txt file
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
