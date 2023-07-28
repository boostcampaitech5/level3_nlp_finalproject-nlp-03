FROM python:3.9

WORKDIR /app

RUN pip install poetry

COPY pyproject.toml .

# Install project dependencies using Poetry
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

# Copy the application code to the working directory in the container (assuming your app code is located in a folder named 'app')
COPY app/ .

RUN alembic revision --autogenerate 
RUN alembic upgrade head
# RUN python testdata.py

# Load DB backup
COPY app.db .

# Expose any necessary ports for your application (if applicable)
EXPOSE 80

# Command to run your application (replace 'main.py' with your actual entry point file)
CMD ["python", "main.py"]