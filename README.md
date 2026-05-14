Proyecto final - Grafos de Transporte Aéreo

-Descripción El presente proyecto consiste en la construcción de un grafo simple, no dirigido y ponderado, el cual modela rutas de transporte aéreo a partir de un conjunto de datos reales.

Cada vértice representa un aeropuerto, mientras que las aristas representan conexiones entre estos, cuyo peso corresponde a la distancia geográfica calculada mediante la fórmula de Haversine.

A partir de este modelo, se implementan diferentes algoritmos fundamentales de teoría de grafos, permitiendo analizar propiedades estructurales del grafo y resolver problemas como:

Conectividad Bipartición Árbol de expansión mínima (MST) Caminos mínimos

-Objetivos Modelar un sistema real utilizando estructuras de datos basadas en grafos. Implementar algoritmos clásicos sin el uso de librerías externas. Analizar propiedades del grafo como conectividad y bipartición. Aplicar algoritmos de optimización como Dijkstra y Kruskal. Integrar una interfaz gráfica para facilitar la interacción con el sistema.

-Dataset Se utiliza el archivo:

data/flights_final.csv Contiene información de vuelos entre aeropuertos, incluyendo:

Código del aeropuerto Nombre Ciudad País Latitud y longitud

-Tecnologías utilizadas Python Streamlit (interfaz gráfica) Pandas (manejo de datos)

-Estructura del proyecto scratch-main/

app.py # Interfaz gráfica (Streamlit)

core/

graph.py # Construcción del grafo

algorithms.py # Algoritmos de grafos

data/

flights_final.csv # Dataset

README.md

-Funcionalidades

Conectividad del grafo Determina si el grafo es conexo
Si no lo es:

Número de componentes Tamaño de cada componente

Verificación de grafo bipartito Determina si el grafo es bipartito
En caso de múltiples componentes:

Evalúa la componente más grande

Árbol de expansión mínima (MST) Se calcula usando el algoritmo de Kruskal
Se obtiene el peso total del árbol

En caso de múltiples componentes:

Se calcula el MST de cada una

Información de aeropuertos Dado un aeropuerto inicial:
Se muestra su información:

-Código

-Nombre

-Ciudad

-País

-Coordenadas

Caminos mínimos Se utiliza el algoritmo de Dijkstra
Se pueden:

Encontrar los 10 aeropuertos más lejanos (por camino mínimo) Mostrar distancias acumuladas

Ruta entre dos aeropuertos Se calcula el camino mínimo entre dos vértices Se muestran los aeropuertos intermedios
-Cálculo de distancias Se utiliza la fórmula de Haversine para calcular la distancia entre dos coordenadas geográficas:

Permite medir distancias sobre la superficie de la Tierra Se usa como peso de las aristas del grafo

-Implementación de algoritmos Todos los algoritmos utilizados en este proyecto fueron implementados manualmente, sin el uso de librerías externas especializadas.

Entre los algoritmos implementados se encuentran:

Búsqueda en anchura (BFS) para conectividad Verificación de bipartición Algoritmo de Kruskal para el árbol de expansión mínima Algoritmo de Dijkstra para caminos mínimos Esto garantiza la comprensión y aplicación directa de los conceptos estudiados en el curso.

-¿Cómo ejecutar el proyecto? Instalar dependencias: pip install streamlit pandas Ejecutar la aplicación: streamlit run app.py Abrir en el navegador: http://localhost:8501
