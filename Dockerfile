# Usa una imagen base que tenga Python y TensorFlow instalados
FROM tensorflow/tensorflow:latest-gpu

# Copia el script al directorio de trabajo en el contenedor
COPY vgg_16.py /usr/src/app/model.py

# Establece el directorio de trabajo
WORKDIR /usr/src/app

RUN pip install scikit-learn
RUN pip install pydot
RUN apt update
RUN apt install graphviz -y 
RUN pip install pillow
RUN pip install pandas
RUN pip install matplotlib
# Ejecuta el script al iniciar el contenedor
CMD ["python", "model.py"]