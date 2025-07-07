FROM electropi_img:latest

WORKDIR /app

COPY . .

RUN python download_models.py

CMD ["streamlit", "run", "app.py"]