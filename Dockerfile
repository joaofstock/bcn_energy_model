FROM python:3.12.7-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all application files into the container
COPY . /app

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the Flask app's port
EXPOSE 5000

# Add an environment variable to suppress Flask warnings and enable production mode
ENV FLASK_ENV=production
ENV FLASK_APP=app.py

# Default command to run the Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
