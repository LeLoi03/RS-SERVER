# Step 1: Use an official, lightweight Python base image
# python:3.11-slim is a good choice as it's smaller than the full Debian image
FROM python:3.11-slim

# Step 2: Set the working directory inside the container
# All subsequent commands will be run from this directory
WORKDIR /app

# Step 3: Copy the requirements file first to leverage Docker's layer caching
# This is a key optimization: if requirements.txt doesn't change, Docker won't
# reinstall all the dependencies on every build, making builds much faster.
COPY requirements.txt .

# Step 4: Install the Python dependencies
# --no-cache-dir keeps the image size smaller
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the rest of your application's code into the container
# This includes your server.py, the 'src' and 'config' directories,
# and most importantly, the 'artifacts' directory with your trained models.
COPY . .

# Step 6: Expose the port the app runs on
# This informs Docker that the container listens on port 8000.
# It's good practice for documentation but doesn't actually publish the port.
EXPOSE 8000

# Step 7: The command to run your application
# We use uvicorn to start the FastAPI server.
# --host 0.0.0.0 is CRITICAL: It tells the server to listen on all available
# network interfaces, making it accessible from outside the container.
# Using the default 127.0.0.1 would only allow connections from within the container itself.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]