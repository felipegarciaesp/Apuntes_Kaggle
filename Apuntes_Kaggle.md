# Intro to Deep Learning

## 1 A Single Neuron

Las redes neuronales se componen de neuronas, donde cada una desempeña individualmente un solo cálculo. El poder de una red neuronal proviene de la complejidad de las conexiones que estas neuronas pueden formar.

Una neurona (o unidad) con un solo input se ve como la siguiente imagen:

![Una sola neurona](https://github.com/felipegarciaesp/Apuntes_Kaggle/blob/main/Imagen_1.jpg)

El input es x, su conexión con la neurona tiene un **peso** que es w. **Una red neuronal aprende por medio de la modificación de sus pesos.**

El sesgo está representado por b y es un tipo especial de peso. **El sesgo permite a la neurona modificar el output indenpendientemente de sus inputs.** El sesgo no tiene ningun input asociado, en el diagrama se pone un 1 para que el valor del sesgo en la neurona sea sólo "b".

Las neuronas pueden tener múltiples inputs, como se muestra en la siguiente imagen:

![Multiples inputs](https://github.com/felipegarciaesp/Apuntes_Kaggle/blob/main/Iimagen_2.jpg)

