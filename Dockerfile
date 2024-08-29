# Use the smallest possible Python base image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /.

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any Python dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

EXPOSE 5000

# Command to run the application
CMD ["streamlit", "src/app.py"]