# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /src/

# Copy the current directory contents into the container at /src
COPY requirements.txt /src/requirements.txt
COPY image_routing_2.png /src/image_routing_2.png

# Install any needed packages specified in requirements.txt
RUN python -m pip install -r requirements.txt
RUN apt-get update
RUN apt-get install poppler-utils -y

# Run app.py when the container launches
CMD ["streamlit", "run", "stream_pgsql01.py"]
