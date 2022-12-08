from __future__ import print_function

import base64
import io
import json
import streamlit as st


import numpy as np
from PIL import Image
import requests
SERVER_URL = 'https://resnet-service-computer-vision-johan-ortega118.cloud.okteto.net/v1/models/resnet:predict'
MODEL_ACCEPT_JPG = False


def main():
    st.title("Clasificador de Imagenes")
    st.subheader('Johan Paul Ortega Murillo')
    #dl_request = requests.get(IMAGE_URL, stream=True)
    #dl_request.raise_for_status()
    img_file_buffer = st.file_uploader(
        "Carga una imagen", type=["png", "jpg", "jpeg"])
    
    if img_file_buffer is not None:
        dl_request = img_file_buffer.read() 
        st.image(np.array(Image.open(img_file_buffer)), caption="Imagen", use_column_width=False)
        if MODEL_ACCEPT_JPG:
            jpeg_bytes = base64.b64encode(dl_request).decode('utf-8')
            predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes
        else:
            #jpeg_rgb = Image.open(io.BytesIO(dl_request))
            # Normalize and batchify the image
            #jpeg_rgb = np.expand_dims(np.array(jpeg_rgb) / 255.0, 0).tolist()
            #predict_request = json.dumps({'instances': jpeg_rgb})
            jpeg_bytes = base64.b64encode(dl_request).decode('utf-8')

            predict_request = '{"instances" : [{"b64": "%s"}]}' % jpeg_bytes
            # base64.b64decode(dl_request.content).decode()


        #image = np.array(Image.open(img_file_buffer))
        #st.image(image, caption="Imagen", use_column_width=False)

    if st.button("Predicción"):
        if img_file_buffer is not None:
            for _ in range(3):
                response = requests.post(SERVER_URL, data=predict_request)
                response.raise_for_status()

            # Send few actual requests and report average latency.
            total_time = 0
            num_requests = 1
            index = 0
            for _ in range(num_requests):
                response = requests.post(SERVER_URL, data=predict_request)
                response.raise_for_status()
                total_time += response.elapsed.total_seconds()
                prediction = response.json()['predictions'][index]['classes']
            ind = "%s" %(prediction-1)
            with open('imagenet_class_index.json') as archivo:
                datos = json.load(archivo)
                nombre = (datos[ind][1])
            #print("El index = %s y el nombre es %s" %ind %nombre)

            st.success('LA CLASE ES: {}'.format(ind) + ' EL NOMBRE ES: {}'.format(nombre))
        else:
            st.error('IMAGEN AÚN NO CARGADA')


if __name__ == '__main__':
    main()
