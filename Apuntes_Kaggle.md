# Intro to Deep Learning

## 1 A Single Neuron

Las redes neuronales se componen de neuronas, donde cada una desempeña individualmente un solo cálculo. El poder de una red neuronal proviene de la complejidad de las conexiones que estas neuronas pueden formar.

Una neurona (o unidad) con un solo input se ve como la siguiente imagen:

![Una sola neurona](https://github.com/felipegarciaesp/Apuntes_Kaggle/blob/main/Imagen_1.jpg)

El input es x, su conexión con la neurona tiene un **peso** que es w. **Una red neuronal aprende por medio de la modificación de sus pesos.**

El sesgo está representado por b y es un tipo especial de peso. **El sesgo permite a la neurona modificar el output indenpendientemente de sus inputs.** El sesgo no tiene ningun input asociado, en el diagrama se pone un 1 para que el valor del sesgo en la neurona sea sólo "b".

Las neuronas pueden tener múltiples inputs, como se muestra en la siguiente imagen:

![Multiples inputs](https://github.com/felipegarciaesp/Apuntes_Kaggle/blob/main/Iimagen_2.jpg)

La fórmula para esta neurona sería: $$y = w_0 x_0 + w_1 x_1 + w_2 x_2 + b$$

La manera más fácil de crear un modelo en Keras es por medio de keras.Sequential, que crea una red neuronal como un conjunto de capas. La sintaxis es la siguiente:

```
from tensorflow import keras
from tensorflow.keras import layers

# Create a network with 1 linear unit
model = keras.Sequential([
    layers.Dense(units=1, input_shape=[3])
])
```

Con units definimos cuántos outputs queremos, en el caso del ejemplo tenemos sólo 1 output.
Con input_shape le decimos a Keras las dimensiones del input. Seteando input_shape = [3] nos aseguramos que el modelo aceptará 3 inputs.

>Why is input_shape a Python list?
>The data we'll use in this course will be tabular data, like in a Pandas dataframe. We'll have one input for each feature in the dataset. The features are arranged by column, so we'll always have input_shape=[num_columns]. The reason Keras uses a list here is to permit use of more complex datasets. Image data, for instance, might need three dimensions: [height, width, channels]

Internamente, Keras representa los pesos de una red neuronal con **tensores**. Los tensores son básicamente una versión de TensorFlow de un numpy array, con algunas diferencias que lo hacen más apto para el deep learning (por ejemplo, los tensores son compatibles con aceleradores GPU y TPU).

Los problemas de **regresion** son problemas de ajuste de curvas, en donde intentamos encontrar la curva que mejor se ajusta a la data.

## 2 Deep Neural Networks

Las redes neuronales tipicamente organizan sus neuronas en **capas** (**layers** en inglés). Cuando juntamos varias neuronas que tienen un set de inputs en común, tenemos una **capa densa** (**dense layer**).