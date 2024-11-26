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

![Capa densa con dos neuronas](https://github.com/felipegarciaesp/Apuntes_Kaggle/blob/main/Imagen%203.jpg)

Se puede pensar en que cada capa de una red neuronal está desempeñando alguna clase de transformación simple. A través de una pila profunda de capas, una red neuronal puede transformar sus inputs de maneras cada vez más complejas.

### La función de activación (The Activation Function)

Dos capas densas con nada entremedio no son mejores que una sola capa densa por sí sola. Las capas densas por sí mismas no podrán movernos nunca del mundo de líneas y planos. Lo que necesitamos es algo *no lineal*, es decir, **funciones de activación**.

![Funcion de activacion](https://github.com/felipegarciaesp/Apuntes_Kaggle/blob/main/Imagen_4.jpg)

Sin funciones de activación, las redes neuronales solo podrían aprender relaciones lineales. Para ajustar curvas, necesitamos funciones de activación.

Una **función de activación** es simplemente una función que se aplican a las salidas de las capas. La más común es la **rectifier function max(0, x)**:

![The rectifier function](https://github.com/felipegarciaesp/Apuntes_Kaggle/blob/main/Imagen_5.jpg)

Cuando aplicamos la funcion rectifier a una unidad lineal, obtendremos una **rectified linear unit** ó **ReLU** (es más común identificar esta función como **función ReLU**). Aplicar una activación ReLU a una unidad lineal significa que como output obtendremos: $$max(0, w*x + b)$$.

### Apilando capas densas

Las capas antes de la capa de salida son denominadas como **ocultas** (**hidden**), ya que nunca vemos sus outputs directamente.

![Creando la red neuronal](https://github.com/felipegarciaesp/Apuntes_Kaggle/blob/main/Imagen_6.jpg)

Fijarse que en la imagen se muestra que la capa final (output) es una unidad lineal (es decir, sin función de activación). Esto hace que esta red sea apropiada para tareas de **regresión**, **donde tratamos de predecir un valor numérico arbitrario**. Otras tareas (como **clasificación**) podrían requerir de una función de activación en el output.

El modelo Sequential utilizado conectará un listado de capas en orden desde el primero hasta el último: la primera capa recibe el/los input(s), y la ultima capa produce el output. El sigueinte código crea el modelo que se muestra en la figura de arriba:

```
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
    # the hidden ReLU layers
    layers.Dense(units=4, activation='relu', input_shape=[2]),
    layers.Dense(units=3, activation='relu'),
    # the linear output layer 
    layers.Dense(units=1),
])
```

La de arriba es la forma usual de definir una función de activación para una capa. Sin embargo, hay situaciones en que se va a requerir la otra forma, correspondiente a la siguiente:

```
layers.Dense(units=8),
layers.Activation('relu')
```

Lo anterior es completamente equivalente a ´layers.Dense(units=8, activation='relu')´.

Asegúrate de pasar todas las capas juntas en una lista, como [layer, layer, layer, ...], en vez de argumentos por separado.

Hay una gran familia de variantes de la activación 'relu' ('elu', 'selu' y 'swish', por nombrar algunas). Algunas podrían desempeñarse mejor que otras, por lo que habría que experimentar cual se adapta mejor en cada caso. La activación ReLU tiende a ser buena en la mayoría de los problemas, por lo que es un buen punto de partida.

## 3 Stochastic Gradient Descent

En esta sección se verá como entrenar una red neuronal, se verá como las redes neuronales aprenden.

Entrenar la red significa ajustar sus pesos de manera tal que pueda transformar, para un set de datos de entrenamiento, los features (inputs) en los target (outputs).

Para entrenar una red neuronal, ademas de tener datos de entrenamiento necesitaremos dos cosas:

- Una **"loss function"** que mida que tan buenas son las predicciones de la red.
- Un **"optimizer"** que le pueda decir a la red como cambiar sus pesos.

### The Loss Function (Función de pérdida)

Durante el entrenamiento, el modelo usará la función de pérdida como una guía para encontrar los valores correctos de sus pesos (menor pérdida es mejor). En otras palabras, la función de pérdida le dice a la red cuál es su objetivo.

Diferentes problemas requerirán diferentes funciones de pérdida. Una función de pérdida común para problemas de regresión es el **error absoluto medio** ó **MAE**. También se encuentran el **error cuadrático medio** (**MSE**) y el **Huber loss**, entre otros.

### El optimizador - Stochastic Gradient Descent, SGD (Gradiente Descendiente Estocástico).

El trabajo del optimizador es decirle a la red como resolver el problema. El optimizador es un algoritmo que ajusta los pesos para minimizar la pérdida. 

Virtualmente, todos los algoritmos utilizados en deep learning pertenecen a una familia llamada **"gradiente descendiente estocástico"**, que son algoritmos iterativos que entrenan a una red en pasos. Un paso de entrenamiento consiste en lo siguiente:

1. Selecciona algunos datos de entrenamiento y los córre en la red para hacer algunas predicciones.
2. Mide la pérdida entre las predicciones y los valores verdaderos.
3. Finalmente, ajusta los pesos de manera que haga la pérdida más pequeña.

Luego, se repiten estos pasos hasta que la pérdida es tan pequeña como sea requerido (o hasta que no pueda decrecer más).

La muestra de cada iteración se denomina **minibatch** (o simplemente **batch**), mientras que una ronda completa de los datos de entrenamiento se denomina **epoch**. El número de epochs que entrenas será la cantidad de veces que la red verá cada ejemplo de entrenamiento.

>NOTA IMPORTANTE:
> - Cuando se entrena una red neuronal, no solo cambian los pesos sino que también el sesgo (bias). Explicado con una regresión lineal, al entrenar la red cambian tanto la pendiente como el intercepto.

### Tasa de aprendizaje y tamaño del batch (Learning Rate and Batch Size)

La tasa de aprendizaje (learning rate) determina el tamaño de los pasos que da el algoritmo para ajustar los pesos de la red durante el proceso de aprendizaje. Un learning rate más pequeño significa que la red necesitará ver más minibatches para que sus pesos convergan a los mejores valores.

> Imagina que estás en una colina tratando de llegar al punto más bajo (mínimo de la función de error). Un learning rate grande sería como dar saltos largos: puedes llegar rápido pero podrías pasarte del mínimo. Un learning rate pequeño es como dar pasos cortos: tardarás más, pero es menos probable que te pases del mínimo.

El learning rate y el tamaño de los minibatches son **los dos parámetros que tienen un mayor efecto en como procede el entrenamiento del SGD**. La elección de valores para estos parámetros no siempre es obvia.

**Adam** es un algoritmo SGD que tiene un learning rate adaptativo que lo hace apto para la mayoría de problemas sin requerir ningún ajuste de parámetros (es "auto-ajustable", en cierto sentido). 

> Adam es un gran optimizador de propósito general.

Luego de definir un modelo, puedes añadir una loss function y un optimizer con el método ´compile´:

```
model.compile(
    optimizer="adam",
    loss="mae",
)
```

>What's In a Name?
>The gradient is a vector that tells us in what direction the weights need to go. More precisely, it tells us how to change the weights to make the loss change fastest. We call our process gradient descent because it uses the gradient to descend the loss curve towards a minimum. Stochastic means "determined by chance." Our training is stochastic because the minibatches are random samples from the dataset. And that's why it's called SGD!

Acá quedé, continuar con EXAMPLE - RED WINE QUALITY.