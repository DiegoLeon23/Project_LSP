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


while (1):
    ret,frame = cap.read()
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    copia = frame.copy()
    resultado = manos.process(color)
    posiciones = []  # En esta lista vamos a almcenar las coordenadas de los puntos
    #print(resultado.multi_hand_landmarks) #Si queremos ver si existe la deteccion

    if resultado.multi_hand_landmarks: #Si hay algo en los resultados entramos al if
        for mano in resultado.multi_hand_landmarks:  #Buscamos la mano dentro de la lista de manos que nos da el descriptor
            for id, lm in enumerate(mano.landmark):  #Vamos a obtener la informacion de cada mano encontrada por el ID
                alto, ancho, c = frame.shape  #Extraemos el ancho y el alto de los fotpgramas para multiplicarlos por la proporcion
                corx, cory = int(lm.x*ancho), int(lm.y*alto) #Extraemos la ubicacion de cada punto que pertence a la mano en coordenadas
                posiciones.append([id,corx,cory])
                dibujo.draw_landmarks(frame, mano, clase_manos.HAND_CONNECTIONS)
            if len(posiciones) != 0:
                pto_i1 = posiciones[3] #5 Dedos: 4 | 0 Dedos: 3 | 1 Dedo: 2 | 2 Dedos: 3 | 3 Dedos: 4 | 4 Dedos: 8
                pto_i2 = posiciones[17]#5 Dedos: 20| 0 Dedos: 17| 1 Dedo: 17| 2 Dedos: 20| 3 Dedos: 20| 4 Dedos: 20
                pto_i3 = posiciones[10]#5 Dedos: 12| 0 Dedos: 10 | 1 Dedo: 20|2 Dedos: 16| 3 Dedos: 12| 4 Dedos: 12
                pto_i4 = posiciones[0] #5 Dedos: 0 | 0 Dedos: 0 | 1 Dedo: 0 | 2 Dedos: 0 | 3 Dedos: 0 | 4 Dedos: 0
                pto_i5 = posiciones[9]
                x1,y1 = (pto_i5[1]-150),(pto_i5[2]-150) #Obtenemos el punto incial y las longitudes
                ancho, alto = (x1+250),(y1+250)
                x2,y2 = x1 + ancho, y1 + alto
                dedos_reg = copia[y1:y2, x1:x2]
                dedos_reg = cv2.resize(dedos_reg, (200, 200), interpolation=cv2.INTER_CUBIC)  # Redimensionamos las fotos
                x = img_to_array(dedos_reg)  # Convertimos la imagen a una matriz
                x = np.expand_dims(x, axis=0)  # Agregamos nuevo eje
                vector = cnn.predict(x)  # Va a ser un arreglo de 2 dimensiones, donde va a poner 1 en la clase que crea correcta
                resultado = vector[0]  # [1,0] | [0, 1]
                respuesta = np.argmax(resultado)  # Nos entrega el indice del valor mas alto 0 | 1
                if respuesta == 0: #A
                    print(vector,resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, '{}'.format(dire_img[0]), (x1, y1 - 5), 1, 1.3, (0, 255, 0), 1, cv2.LINE_AA)
                elif respuesta == 1:#B
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, '{}'.format(dire_img[1]), (x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
                elif respuesta == 2:#C
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, '{}'.format(dire_img[2]), (x1, y1 - 5), 1, 1.3, (0, 255, 255), 1, cv2.LINE_AA)
                elif respuesta == 3:#D
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(frame, '{}'.format(dire_img[3]), (x1, y1 - 5), 1, 1.3, (255, 0, 255), 1, cv2.LINE_AA)
                elif respuesta == 4:#E
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, '{}'.format(dire_img[4]), (x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
                elif respuesta == 5:#F
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, '{}'.format(dire_img[5]), (x1, y1 - 5), 1, 1.3, (0, 255, 255), 1, cv2.LINE_AA)
                elif respuesta == 6:#G
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, '{}'.format(dire_img[6]), (x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
                elif respuesta == 7:#H
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(frame, '{}'.format(dire_img[7]), (x1, y1 - 5), 1, 1.3, (255, 0, 255), 1, cv2.LINE_AA)
                elif respuesta == 8:#I
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, '{}'.format(dire_img[8]), (x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
                elif respuesta == 9:#K
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, '{}'.format(dire_img[9]), (x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
                elif respuesta == 10:#L
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, '{}'.format(dire_img[10]), (x1, y1 - 5), 1, 1.3, (0, 255, 255), 1, cv2.LINE_AA)
                elif respuesta == 11:#M
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    cv2.putText(frame, '{}'.format(dire_img[11]), (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif respuesta == 12:#N
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, '{}'.format(dire_img[12]), (x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
                elif respuesta == 13:#O
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, '{}'.format(dire_img[13]), (x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
                elif respuesta == 14:#P
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, '{}'.format(dire_img[14]), (x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
                elif respuesta == 15:#Q
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    cv2.putText(frame, '{}'.format(dire_img[15]), (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif respuesta == 16:#R
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 3)
                    cv2.putText(frame, '{}'.format(dire_img[16]), (x1, y1 - 5), 1, 1.3, (255, 0, 0), 1, cv2.LINE_AA)
                elif respuesta == 17:#S
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, '{}'.format(dire_img[17]), (x1, y1 - 5), 1, 1.3, (0, 255, 0), 1, cv2.LINE_AA)
                elif respuesta == 18:#T
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(frame, '{}'.format(dire_img[18]), (x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
                elif respuesta == 19:#U
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
                    cv2.putText(frame, '{}'.format(dire_img[19]), (x1, y1 - 5), 1, 1.3, 0, 255, 255), 1, cv2.LINE_AA)
                elif respuesta == 20:#V
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, '{}'.format(dire_img[20]), (x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
                elif respuesta == 21:#W
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(frame, '{}'.format(dire_img[21]), (x1, y1 - 5), 1, 1.3, (255, 0, 255), 1, cv2.LINE_AA)
                elif respuesta == 22:#X
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(frame, '{}'.format(dire_img[22]), (x1, y1 - 5), 1, 1.3, (0, 0, 255), 1, cv2.LINE_AA)
                elif respuesta == 23:#Y
                    print(vector, resultado)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    cv2.putText(frame, '{}'.format(dire_img[23]), (x1, y1 - 5), 1, 1.3, (255, 0, 255), 1, cv2.LINE_AA)
                else:
                    cv2.putText(frame, 'LETRA DESCONOCIDA', (x1, y1 - 5), 1, 1.3, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow("Video",frame)
    k = cv2.waitKey(1)
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()











