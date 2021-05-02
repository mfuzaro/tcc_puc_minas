# -*- coding: utf-8 -*-
"""
@author: Marcos Fuzaro Junior
"""

# =============================================================================
# Importanto as bibliotecas
# =============================================================================
import os
import cv2 as cv
from sklearn.neighbors import KNeighborsClassifier

# =============================================================================
# Definindo funcoes
# =============================================================================
def get_image_features_BGR(img):
    """
    Funcao que recebe uma imagem com 4 canais de cor
    Extrai as caracteristicas de cor (media de cada canal)
    Remove o ultimo elemento, pois o sistema de cor RGB utiliza 3 canais
    Retorna uma lista com a media dos valores de cada canal RGB (ordem BGR)
    """
    mean_value_BGR = cv.mean(img)
    mean_value_BGR = list(mean_value_BGR)
    mean_value_BGR.pop(len(mean_value_BGR)-1)
    return mean_value_BGR

# =============================================================================
# Definindo os caminhos dos diretorios
# =============================================================================
dataset_train_path = "images\\datasets\\train\\"
dataset_test_path = "images\\datasets\\test\\"
dataset_classified_path = "images\\datasets\\classified\\"

# =============================================================================
# Percorrendo o diretorio das imagens de treinamento
# Extraindo as caracteristicas e classes
# =============================================================================

# Lista de listas, em que cada lista interna representa as características 
# de uma imagem. Exemplo de formato [[B, G, R], [B, G, R]]
images_features = []

# Classes, em que cada elemento representa a respectiva classe de uma imagem
# Por exemplo, images_targets[0] representa a classe da 
# primeira imagem, em images_features[0]
images_targets = []

dirs = os.listdir(dataset_train_path)

# Variavel utilizada para guardar os nomes das classes
targets_names = dirs.copy()

for d in dirs:
    files = os.listdir(dataset_train_path + "\\" + d)
    for f in files:
        path_image = dataset_train_path + "\\" + d + "\\" + f
        # Lendo a imagem
        img_BGR = cv.imread(path_image, cv.IMREAD_COLOR)
        # Extraindo as caracteristicas e classes
        images_features.append(get_image_features_BGR(img_BGR))
        images_targets.append(d)

print("Numero de imagens de treinamento: {}".format(len(images_features)))

print("\nNumero de classes: {}".format(len(targets_names)))

print("\nClasses: \n{}".format(targets_names))

# =============================================================================
# Percorrendo o diretorio de imagens de teste
# Extraindo as caracteristicas
# =============================================================================

# Lista de listas, em que cada lista interna representa as características 
# de uma imagem de teste. Exemplo de formato [[B, G, R], [B, G, R]]
images_test_features = []

dirs = os.listdir(dataset_test_path)

# Lista de imagens de teste
images_test = []

for d in dirs:
    files = os.listdir(dataset_test_path + "\\" + d)
    for f in files:
        path_image = dataset_test_path + "\\" + d + "\\" + f
        # Lendo a imagem
        img_test = cv.imread(path_image, cv.IMREAD_COLOR)
        # Guardando a imagem para ser escrita posteriormente no diretorio
        images_test.append(img_test)
        # Extraindo as caracteristicas
        images_test_features.append(get_image_features_BGR(img_test))

print("\nNumero de imagens de teste: {}".format(len(images_test_features)))

# =============================================================================
# Classificacao das imagens com K-NN
# =============================================================================

# Definindo o numero de vizinhos
knn	=	KNeighborsClassifier(3)

# Treinando o classificador
knn.fit(images_features, images_targets)

# Nivel de acuracia media
score = knn.score(images_features, images_targets)
print("\nAcuracia media: {}".format(score))

# Classificando a lista de imagens
classified_labels = knn.predict(images_test_features)
print("\nNumero de imagens classificadas: {}".format(len(classified_labels)))

# =============================================================================
# Salvando em diretorios as imagens de teste conforme a classificacao
# =============================================================================

# Formato da imagem
fmt = ".jpg"

image_index = 0

# Criando um diretorio para cada classe
for t in targets_names:
    dir_path = dataset_classified_path + t
    # Se o diretorio nao existe, entao e criado
    if not(os.path.isdir(dir_path)):
        os.mkdir(dir_path)

# Salvando as imagens nos diretorios de acordo com a classe atribuida
for label in classified_labels:
    cv.imwrite(dataset_classified_path + label + "\\" + str(image_index) + fmt, 
                images_test[image_index])
    image_index += 1








