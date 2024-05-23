import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Cargar el modelo entrenado y las clases
model = tf.keras.models.load_model('best_model.keras')
class_names = np.loadtxt('clases.txt', dtype=str).tolist()

# Parámetros de la imagen
img_height = 180
img_width = 180

# Función para cargar y preprocesar una imagen
def load_and_preprocess_image(image_path, img_height, img_width):
    img = Image.open(image_path).resize((img_height, img_width))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)  # Crear un batch
    return img_array

# Función para abrir la cámara y tomar una foto
def take_photo():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Presiona Espacio para tomar una foto", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error al abrir la cámara")
            break
        cv2.imshow("Presiona Espacio para tomar una foto", frame)
        if cv2.waitKey(1) & 0xFF == ord(' '):
            img_name = "captured_image.jpg"
            cv2.imwrite(img_name, frame)
            st.write(f"Foto guardada como {img_name}")
            break

    cap.release()
    cv2.destroyAllWindows()
    return img_name

def main():
    st.set_page_config(page_title="Proyecto IA - Leandro Cortez", layout="wide")

    # Título y subtítulo
    st.title("Proyecto IA - Leandro Cortez")
    st.subheader("Reconocimiento de Productos de Supermercado")

    # Información en la barra lateral izquierda
    st.sidebar.title("Información del Proyecto")
    st.sidebar.info("""
        **Nombre:** Leandro Cortez
        **Proyecto:** IA para Reconocimiento de Productos de Supermercado
        **Productos Reconocidos:**
        - Leche / MILK
        - Banana 
        - Avocado 
        - Manzana / Apple
        - Gaseosas / Sodas
        - Paquetes de papas / Chips
        - Galletas / Biscuits
    """)

    st.sidebar.title("¿Cómo funciona?")
    st.sidebar.info("""
        - Puedes subir una imagen desde tu dispositivo o tomar una foto con tu cámara.
        - El modelo de IA analizará la imagen y predecirá a qué clase de producto pertenece.
    """)

    # Layout para incluir una barra lateral derecha
    col1, col2 = st.columns([3, 1])

    with col1:
        # Opción para subir una imagen o tomar una foto
        option = st.radio("Selecciona una opción", ('Subir imagen', 'Tomar foto'))

        if option == 'Subir imagen':
            uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                # Preprocesar la imagen
                img_array = load_and_preprocess_image(uploaded_file, img_height, img_width)

                # Realizar la predicción
                predictions = model.predict(img_array)
                score = tf.nn.softmax(predictions[0])

                # Mostrar la imagen y la predicción
                img = Image.open(uploaded_file)
                st.image(img, caption="Imagen subida", use_column_width=True)
                st.write(
                    "Esta imagen pertenece a la clase {} con una confianza de {:.2f} %."
                    .format(class_names[np.argmax(score)], 100 * np.max(score))
                )
        elif option == 'Tomar foto':
            if st.button('Abrir cámara'):
                file_path = take_photo()
                if file_path:
                    # Preprocesar la imagen
                    img_array = load_and_preprocess_image(file_path, img_height, img_width)

                    # Realizar la predicción
                    predictions = model.predict(img_array)
                    score = tf.nn.softmax(predictions[0])

                    # Mostrar la imagen y la predicción
                    img = Image.open(file_path)
                    st.image(img, caption="Foto tomada", use_column_width=True)
                    st.write(
                        "Esta imagen pertenece a la clase {} ."
                        .format(class_names[np.argmax(score)], 100 * np.max(score))
                    )

    with col2:
        st.title("Información adicional")
        st.image("supermercado.jpg", caption="Supermercado", use_column_width=True)
        st.write("""
        **Fuente de las imágenes:**
        - Google Images
        - Kaggle Datasets
        - Roboflow
        - Imagenes propias
        """)

if __name__ == "__main__":
    main()
