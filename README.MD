# Análisis de Sentimientos en Tweets en Español usando Modelos de NLP

Este repositorio contiene un notebook de Jupyter (`NLP_PHAWAI.ipynb`) que explora y compara tres enfoques diferentes para la clasificación de emociones en tweets en español. El objetivo es analizar el rendimiento de un modelo clásico de Machine Learning, un modelo de Transformer con fine-tuning y un Large Language Model (LLM) en una tarea de PNL.

## 📝 Tabla de Contenidos
* [Descripción del Proyecto](#-descripción-del-proyecto)
* [Modelos Implementados](#-modelos-implementados)
* [Dataset](#-dataset)
* [Metodología](#️-metodología)
* [Resultados y Conclusiones](#-resultados-y-conclusiones)
* [Tecnologías Utilizadas](#️-tecnologías-utilizadas)
* [Cómo Ejecutar el Notebook](#-cómo-ejecutar-el-notebook)
* [Autor](#-autor)

## 📖 Descripción del Proyecto

El proyecto aborda la tarea de clasificación de emociones en texto. Se utiliza el dataset **EmoEvent**, que contiene tweets en español asociados a diferentes eventos y etiquetados con una de las siete emociones: `anger`, `disgust`, `fear`, `joy`, `sadness`, `surprise` y `others`.

Se implementan y evalúan los siguientes tres modelos:
1.  **Modelo Clásico:** Regresión Logística con vectorización TF-IDF como línea base.
2.  **Modelo Transformer:** Fine-tuning de **BETO**, una versión de BERT pre-entrenada en español.
3.  **Modelo Generativo (LLM):** **Gemini 2.5 Flash** a través de su API, utilizando técnicas de *zero-shot prompting* (con y sin ejemplos).

## 🤖 Modelos Implementados

### 1. Regresión Logística con TF-IDF
Este es el modelo de referencia. Utiliza `scikit-learn` para convertir el texto de los tweets en vectores numéricos usando la técnica TF-IDF (Term Frequency-Inverse Document Frequency) y luego entrena un clasificador de Regresión Logística.
- **Librerías:** `Pandas`, `Scikit-learn`.
- **Resultado:** Alcanzó una precisión (accuracy) aproximada del **29%**.

### 2. Fine-Tuning de BETO
Este enfoque utiliza un modelo Transformer pre-entrenado específicamente para español ("BETO"). Se realiza un proceso de fine-tuning sobre el dataset de EmoEvent para adaptar el modelo a la tarea específica de clasificación de emociones.
- **Librerías:** `Hugging Face Transformers`, `PyTorch`.
- **Resultado:** Mostró un rendimiento significativamente superior, con una precisión por evento que llegó hasta el **46%** en la categoría `LaLiga`.

### 3. Gemini 2.5 Flash (Zero-Shot)
Se utiliza el LLM de Google, Gemini, para clasificar los tweets sin un entrenamiento previo (zero-shot). Se probaron dos estrategias de prompting:
- **Prompt sin ejemplos:** Solo se le indica al modelo la tarea y las posibles etiquetas.
- **Prompt con ejemplos (Few-shot):** Se le proporciona al modelo un ejemplo claro para cada una de las siete emociones antes de pedirle que clasifique el nuevo tweet.
- **Librerías:** `google-generativeai`.
- **Resultado:** El prompt con ejemplos obtuvo una precisión del **43%** en una muestra de 200 tweets, demostrando la importancia del "prompt engineering". El prompt sin ejemplos tuvo un rendimiento considerablemente menor (18%).

## 📊 Dataset

- **Nombre:** EmoEvent
- **Idioma:** Español (es)
- **Contenido:** Tweets relacionados con eventos específicos (ej. `LaLiga`, `NotreDame`, `Venezuela`).
- **Etiquetas:** `anger`, `disgust`, `fear`, `joy`, `others`, `sadness`, `surprise`.
- **Fuente:** Los datos se cargan directamente desde el repositorio de GitHub: `fmplaza/EmoEvent`.

## ⚙️ Metodología

El flujo de trabajo seguido en el notebook es el siguiente:

1.  **Carga de Datos:** Se cargan los conjuntos de entrenamiento, validación y prueba desde archivos TSV.
2.  **Limpieza de Texto:** Se aplica una función de preprocesamiento para:
    - Eliminar URLs, menciones de usuario y hashtags.
    - Quitar caracteres especiales y números.
    - Convertir el texto a minúsculas.
    - Eliminar espacios extra.
3.  **Codificación de Etiquetas:** Las etiquetas de texto (emociones) se convierten a formato numérico.
4.  **Entrenamiento y Evaluación:** Cada uno de los tres modelos se entrena (o se utiliza para inferencia, en el caso de Gemini) y se evalúa utilizando métricas como el reporte de clasificación y la matriz de confusión.
5.  **Análisis de Resultados:** Se comparan las métricas de rendimiento y se analizan los errores y aciertos, incluyendo un desglose de la precisión por evento para el modelo BETO.

## 📈 Resultados y Conclusiones

- El modelo **BETO con fine-tuning** fue el que obtuvo el mejor rendimiento general, demostrando la eficacia de los modelos Transformer para tareas de PNL en español cuando se adaptan a un dominio específico.
- La **Regresión Logística** sirvió como una línea base útil, pero su rendimiento fue limitado debido a que no captura el contexto semántico de la misma manera que los modelos más avanzados.
- **Gemini 2.5 Flash** mostró un gran potencial, especialmente al utilizar **prompts con ejemplos**, lo que subraya la importancia de guiar adecuadamente a los LLMs para obtener resultados precisos en tareas de *zero-shot*.

## 🛠️ Tecnologías Utilizadas

- **Lenguaje:** Python 3
- **Librerías Principales:**
    - `Pandas` y `NumPy` para manipulación de datos.
    - `Scikit-learn` para el modelo de Regresión Logística y métricas de evaluación.
    - `Matplotlib` y `Seaborn` para visualización de datos.
    - `Hugging Face Transformers` para el fine-tuning del modelo BETO.
    - `google-generativeai` para interactuar con la API de Gemini.
- **Entorno:** Google Colab con aceleración por GPU.

## 🚀 Cómo Ejecutar el Notebook

Para ejecutar este notebook y replicar los resultados, sigue estos pasos:

1.  **Clona el repositorio:**
    ```bash
    git clone [https://github.com/tu-usuario/tu-repositorio.git](https://github.com/tu-usuario/tu-repositorio.git)
    cd tu-repositorio
    ```

2.  **Instala las dependencias:**
    Abre el notebook y ejecuta la celda que contiene el siguiente comando:
    ```bash
    !pip install pandas numpy scikit-learn matplotlib seaborn transformers datasets google-generativeai
    ```
    *Nota: `torch` y otras dependencias de `transformers` pueden ser necesarias.*

3.  **Configura la API Key (para el modelo Gemini):**
    - Obtén una API Key desde [Google AI Studio](https://aistudio.google.com/app/apikey).
    - En Google Colab, ve al panel de la izquierda, haz clic en el ícono de la llave (Secretos) y crea un nuevo secreto llamado `GOOGLE_API_KEY` con tu clave como valor.
    - La celda `userdata.get('GOOGLE_API_KEY')` cargará la clave de forma segura.

4.  **Ejecuta las celdas:**
    - Se recomienda utilizar un entorno de ejecución con **GPU** en Google Colab para acelerar el fine-tuning del modelo BETO. Puedes cambiarlo en `Entorno de ejecución` > `Cambiar tipo de entorno de ejecución`.
    - Ejecuta todas las celdas del notebook en orden.

## ✨ Autor

- **Elmer Andrés Collanqui Casapia**
- 💻 Ingeniería en Sistemas e Informática
- 🚀 Aimara Lab
- 📍 Universidad Nacional de Moquegua
- ✉️ **Contacto:** [ecollanqui@aimaralab.com](mailto:ecollanqui@aimaralab.com)
- 🌐 [GitHub](https://github.com/Mermas-CC) | [LinkedIn](https://www.linkedin.com/in/elmer-andres-collanqui-casapia-977325315)
