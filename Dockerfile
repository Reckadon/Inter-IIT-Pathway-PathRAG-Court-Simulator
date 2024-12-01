FROM pathwaycom/pathway:latest

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire package
COPY . .

# Install the package in development mode
# RUN pip install -e .

# Expose port
EXPOSE 8000

# Command to run the API server
CMD ["python", "-m", "app"]
