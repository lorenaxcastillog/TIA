# Clasificación de mamografías utilizando Redes Convolucionales(CNN)

Proyecto para Tópicos de Inteligencia Artificial - Ciencia de la Computación - UNSA 

## Resumen

El cáncer de mamas produce una tasa alta de mortalidad a lo largo del mundo. El diagnóstico temprano es esencial para el tratamiento, sin embargo es difícil analizar los tejidos de alta densidad. Sistemas de diagnóstico computarizados han sido propuestos para clasificar la densidad de las mamografías, teniendo como desafío mayor, definir características que representen de mejor manera las imágenes a ser clasificadas.

En este trabajo, se replicó el trabajo de **W. R. Silva, Classification of Mammograms by the Breast**
**Composition.**(https://www.researchgate.net/publication/265798808_Classification_of_Mammograms_by_the_Breast_Composition) utilizando C++ y redes convolucionales.

### Prerequisitos

La base de datos utilizada fue MIAS Database(http://peipa.essex.ac.uk/info/mias.html)


## Implementado con

* Qt-Creator
* C++
* [tinny-dnn](https://github.com/tiny-dnn/tiny-dnn)

### CNN
```
Arquitectura:

```
Configuración de la red:

```
nn << convolutional_layer(48, 48, 5, 1,
                                      16,  
                                      padding::valid, true, 1, 1, backend_type)
               << relu_layer(44, 44, 16)
               << max_pooling_layer(44, 44, 16,
                                        2)  
               << relu_layer(22, 22, 16)

               << fully_connected_layer(7744, 3, true,   
                                        backend_type);
```
## Integrantes

* **Grimaldo Dávila** 
* **Lorena Castillo**
  
