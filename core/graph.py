"""
Módulo graph.py

Este módulo contiene la estructura principal del grafo de rutas aéreas.
El grafo se construye a partir del archivo flights_final.csv, donde cada
vértice representa un aeropuerto y cada arista representa una ruta entre dos
aeropuertos.

El grafo es:
- Simple: no se guardan aristas repetidas entre el mismo par de aeropuertos.
- No dirigido: si existe una ruta A-B, también se considera B-A.
- Ponderado: el peso de cada arista es la distancia geográfica entre aeropuertos.
"""

import csv
import math

# Haversine formula to calculate the distance between two geographical points
def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calcula la distancia aproximada en kilómetros entre dos puntos geográficos.

    Se utiliza la fórmula de Haversine, que permite estimar la distancia entre
    dos coordenadas sobre la superficie de la Tierra usando latitud y longitud.

    Parámetros:
        lat1 (float): Latitud del primer punto.
        lon1 (float): Longitud del primer punto.
        lat2 (float): Latitud del segundo punto.
        lon2 (float): Longitud del segundo punto.

    Retorna:
        float: Distancia aproximada entre los dos puntos en kilómetros.
    """
    R = 6371.0 # Radio de la Tierra en kilómetros
    
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2.0)**2 + \
        math.cos(phi1) * math.cos(phi2) * \
        math.sin(delta_lambda / 2.0)**2
        
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    distance = R * c
    return distance

class FlightGraph:
    """
    Representa un grafo de aeropuertos y rutas aéreas.

    Atributos:
        adj_list (dict): Lista de adyacencia del grafo. La estructura es:
            {
                'BOG': {'MDE': peso, 'CLO': peso},
                ...
            }

        airports (dict): Diccionario con la información de cada aeropuerto.
            La llave es el código del aeropuerto y el valor contiene sus datos.
    """
    def __init__(self):
        """
        Inicializa un grafo vacío.

        Se crean dos diccionarios:
        - adj_list: guarda las conexiones entre aeropuertos.
        - airports: guarda la información descriptiva de cada aeropuerto.
        """
        # self.adj_list: dict[str, dict[str, float]]
        # Mapa de código de aeropuerto a un diccionario de destinos y pesos
        self.adj_list = {}
        
        # self.airports: dict[str, dict]
        # Almacena metadatos de los aeropuertos (Nombre, Ciudad, País, Latitud, Longitud)
        self.airports = {}
        
    def add_airport(self, code, name, city, country, lat, lon):
        """
        Agrega un aeropuerto al grafo si aún no existe.

        Parámetros:
            code (str): Código del aeropuerto.
            name (str): Nombre del aeropuerto.
            city (str): Ciudad donde se encuentra el aeropuerto.
            country (str): País donde se encuentra el aeropuerto.
            lat (float | str): Latitud geográfica.
            lon (float | str): Longitud geográfica.

        Retorna:
            None
        """
        if code not in self.airports:
            self.airports[code] = {
                'code': code,
                'name': name,
                'city': city,
                'country': country,
                'lat': float(lat),
                'lon': float(lon)
            }
            if code not in self.adj_list:
                self.adj_list[code] = {}

    def add_edge(self, source, dest, weight):
        """
        Agrega una arista no dirigida entre dos aeropuertos.

        Como el grafo es no dirigido, se registra la conexión en ambos sentidos:
        source -> dest y dest -> source.

        Parámetros:
            source (str): Código del aeropuerto origen.
            dest (str): Código del aeropuerto destino.
            weight (float): Peso de la arista, correspondiente a la distancia.

        Retorna:
            None
        """
        # Es un grafo no dirigido simple
        if dest not in self.adj_list[source]:
            self.adj_list[source][dest] = weight
        if source not in self.adj_list[dest]:
            self.adj_list[dest][source] = weight

    def load_from_csv(self, filepath):
        """
        Carga los datos de aeropuertos y rutas desde un archivo CSV.

        Por cada fila del dataset:
        1. Se extrae el aeropuerto origen.
        2. Se extrae el aeropuerto destino.
        3. Se calcula la distancia entre ambos usando Haversine.
        4. Se agrega la arista al grafo si no existe previamente.

        Parámetros:
            filepath (str): Ruta del archivo CSV que contiene los vuelos.

        Retorna:
            None
        """
        with open(filepath, mode='r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                # Extraemos Aeropuerto Origen
                src_code = row['Source Airport Code']
                src_lat = float(row['Source Airport Latitude'])
                src_lon = float(row['Source Airport Longitude'])
                self.add_airport(
                    code=src_code,
                    name=row['Source Airport Name'],
                    city=row['Source Airport City'],
                    country=row['Source Airport Country'],
                    lat=src_lat,
                    lon=src_lon
                )
                
                # Extraemos Aeropuerto Destino
                dst_code = row['Destination Airport Code']
                dst_lat = float(row['Destination Airport Latitude'])
                dst_lon = float(row['Destination Airport Longitude'])
                self.add_airport(
                    code=dst_code,
                    name=row['Destination Airport Name'],
                    city=row['Destination Airport City'],
                    country=row['Destination Airport Country'],
                    lat=dst_lat,
                    lon=dst_lon
                )
                
                # Calculamos el peso (distancia) solo si no existe la misma arista con otro peso
                # Es grafo simple, por lo que una arista por par de vértices es suficiente
                if dst_code not in self.adj_list[src_code]:
                    distance = haversine_distance(src_lat, src_lon, dst_lat, dst_lon)
                    self.add_edge(src_code, dst_code, distance)

    def get_vertices(self):
        """
        Obtiene todos los vértices del grafo.

        Retorna:
            list: Lista con los códigos de todos los aeropuertos registrados.
        """
        return list(self.adj_list.keys())

    def get_edges(self):
        """
        Obtiene todas las aristas del grafo sin repetirlas.

        Como el grafo es no dirigido, una conexión A-B aparece internamente como
        A -> B y B -> A. Por eso se usa un conjunto auxiliar para evitar duplicados.

        Retorna:
            list: Lista de tuplas con formato (origen, destino, peso).
        """
        # Retorna todas las aristas (u, v, peso) evitando duplicados
        edges = []
        visited = set()
        for u in self.adj_list:
            for v, weight in self.adj_list[u].items():
                # Para grafo no dirigido, (u, v) es igual a (v, u)
                edge = tuple(sorted([u, v]))
                if edge not in visited:
                    visited.add(edge)
                    edges.append((edge[0], edge[1], weight))
        return edges
