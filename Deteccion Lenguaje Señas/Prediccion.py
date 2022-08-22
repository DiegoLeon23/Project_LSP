import cv2
import mediapipe as mp
import os
import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from keras.models import load_model

modelo = 'E:/Documentos/UNMSM/CICLO IX/SOFTWARE INTELIGENTE/data/Modelo.h5'
peso =  'E:/Documentos/UNMSM/CICLO IX/SOFTWARE INTELIGENTE/data/pesos.h5'
cnn = load_model(modelo)  #Cargamos el modelo
cnn.load_weights(peso)  #Cargamos los pesos

direccion = 'E:/Documentos/UNMSM/CICLO IX/SOFTWARE INTELIGENTE/data/Validacion'
dire_img = os.listdir(direccion)
print("Nombres: ", dire_img)

#Leemos la camara
cap = cv2.VideoCapture(0)

#----------------------------Creamos un obejto que va almacenar la deteccion y el seguimiento de las manos------------
clase_manos  =  mp.solutions.hands
manos = clase_manos.Hands() #Primer parametro, FALSE para que no haga la deteccion 24/7
                            #Solo hara deteccion cuando hay una confianza alta
                            #Segundo parametro: numero maximo de manos
                            #Tercer parametro: confianza minima de deteccion
                            #Cuarto parametro: confianza minima de seguimiento

#----------------------------------Metodo para dibujar las manos---------------------------
dibujo = mp.solutions.drawing_utils #Con este metodo dibujamos 21 puntos criticos de la mano

min = 50
max = 300
upper_left = (min, min)
bottom_right = (max, max)
upper_left_rectangle = (min-1, min-1)
bottom_right_rectangle = (max+1, max+1)

while (1):
    ret,frame = cap.read()

    cv2.rectangle(frame, upper_left_rectangle, bottom_right_rectangle, (100, 50, 200), 3)

    dedos_reg = frame[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]
    dedos_reg = cv2.resize(dedos_reg, (200, 200), interpolation=cv2.INTER_CUBIC)  # Redimensionamos las fotos
    x = img_to_array(dedos_reg)  # Convertimos la imagen a una matriz
    x = np.expand_dims(x, axis=0)  # Agregamos nuevo eje
    vector = cnn.predict(x)  # Va a ser un arreglo de 2 dimensiones, donde va a poner 1 en la clase que crea correcta
    resultado = vector[0]  # [1,0] | [0, 1]
    respuesta = np.argmax(resultado)  # Nos entrega el indice del valor mas alto 0 | 1

    if respuesta == 0: #A
        cv2.putText(frame, '{}'.format(dire_img[0]), (upper_left[1], bottom_right[0] - 5), 1, 1.3, (0, 255, 0), 1, cv2.LINE_AA)
    elif respuesta == 1:#B
        cv2.putText(frame, '{}'.format(dire_img[1]), (upper_left[1], bottom_right[0] - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
    elif respuesta == 2:#C
        cv2.putText(frame, '{}'.format(dire_img[2]), (upper_left[1], bottom_right[0] - 5), 1, 1.3, (0, 255, 255), 1, cv2.LINE_AA)
    elif respuesta == 3:#D
        cv2.putText(frame, '{}'.format(dire_img[3]), (upper_left[1], bottom_right[0] - 5), 1, 1.3, (255, 0, 255), 1, cv2.LINE_AA)
    elif respuesta == 4:#E
        cv2.putText(frame, '{}'.format(dire_img[4]), (upper_left[1], bottom_right[0] - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
    elif respuesta == 5:#F
        cv2.putText(frame, '{}'.format(dire_img[5]), (upper_left[1], bottom_right[0] - 5), 1, 1.3, (0, 255, 255), 1, cv2.LINE_AA)
    elif respuesta == 6:#G
        cv2.putText(frame, '{}'.format(dire_img[6]), (upper_left[1], bottom_right[0] - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
    elif respuesta == 7:#H
        cv2.putText(frame, '{}'.format(dire_img[7]), (upper_left[1], bottom_right[0] - 5), 1, 1.3, (255, 0, 255), 1, cv2.LINE_AA)
    elif respuesta == 8:#I
        cv2.putText(frame, '{}'.format(dire_img[8]), (upper_left[1], bottom_right[0] - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
    elif respuesta == 9:#K
        cv2.putText(frame, '{}'.format(dire_img[9]), (upper_left[1], bottom_right[0] - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
    elif respuesta == 10:#L
        cv2.putText(frame, '{}'.format(dire_img[10]), (upper_left[1], bottom_right[0] - 5), 1, 1.3, (0, 255, 255), 1, cv2.LINE_AA)
    elif respuesta == 11:#M
        cv2.putText(frame, '{}'.format(dire_img[11]), (upper_left[1], bottom_right[0] - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
    elif respuesta == 12:#N
        cv2.putText(frame, '{}'.format(dire_img[12]), (upper_left[1], bottom_right[0] - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
    elif respuesta == 13:#O
        cv2.putText(frame, '{}'.format(dire_img[13]), (upper_left[1], bottom_right[0] - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
    elif respuesta == 14:#P
        cv2.putText(frame, '{}'.format(dire_img[14]), (upper_left[1], bottom_right[0] - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
    elif respuesta == 15:#Q
        cv2.putText(frame, '{}'.format(dire_img[15]), (upper_left[1], bottom_right[0] - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
    elif respuesta == 16:#R
        cv2.putText(frame, '{}'.format(dire_img[16]), (upper_left[1], bottom_right[0] - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
    elif respuesta == 17:#S
        cv2.putText(frame, '{}'.format(dire_img[17]), (upper_left[1], bottom_right[0] - 5), 1, 1.3, (0, 255, 0), 1, cv2.LINE_AA)
    elif respuesta == 18:#T
        cv2.putText(frame, '{}'.format(dire_img[18]), (upper_left[1], bottom_right[0] - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
    elif respuesta == 19:#U
        cv2.putText(frame, '{}'.format(dire_img[19]), (upper_left[1], bottom_right[0] - 5), 1, 1.3, 0, 255, 255), 1, cv2.LINE_AA)
    elif respuesta == 20:#V
        cv2.putText(frame, '{}'.format(dire_img[20]), (upper_left[1], bottom_right[0] - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
    elif respuesta == 21:#W
        cv2.putText(frame, '{}'.format(dire_img[21]), (upper_left[1], bottom_right[0] - 5), 1, 1.3, (255, 0, 255), 1, cv2.LINE_AA)
    elif respuesta == 22:#X
        cv2.putText(frame, '{}'.format(dire_img[22]), (upper_left[1], bottom_right[0] - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
    elif respuesta == 23:#Y
        cv2.putText(frame, '{}'.format(dire_img[23]), (upper_left[1], bottom_right[0] - 5), 1, 1.3, (255, 0, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'LETRA DESCONOCIDA', (upper_left[1], bottom_right[0] - 5), 1, 1.3, (0, 255, 255), 1, cv2.LINE_AA)

    print(vector, resultado)

    cv2.imshow("Video",frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
    
cap.release()
cv2.destroyAllWindows()











