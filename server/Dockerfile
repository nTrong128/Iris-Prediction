FROM python:3.12

# 2
RUN pip install Flask gunicorn joblib scikit-learn pandas numpy

# 3
COPY src/ /app
WORKDIR /app

# 4
ENV PORT 8080

# 5
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
