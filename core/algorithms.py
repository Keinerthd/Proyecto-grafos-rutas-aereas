"""
Módulo algorithms.py

Este módulo contiene los algoritmos principales aplicados sobre el grafo de
rutas aéreas. Los algoritmos fueron implementados manualmente para cumplir con
las restricciones del laboratorio.

Algoritmos incluidos:
- BFS para componentes conexas.
- Verificación de grafo bipartito mediante coloreo.
- Kruskal para árbol de expansión mínima.
- Dijkstra para caminos mínimos.
- Reconstrucción del camino mínimo.
"""

import heapq

def find_components(graph):
    """
    Encuentra las componentes conexas del grafo usando búsqueda en anchura (BFS).

    Una componente conexa es un conjunto de vértices donde existe al menos un
    camino entre cualquier par de vértices pertenecientes a esa componente.

    Parámetros:
        graph (FlightGraph): Grafo sobre el cual se hará la búsqueda.

    Retorna:
        list: Lista de componentes. Cada componente es una lista de códigos de aeropuertos.
    """
    visited = set()
    components = []
    
    vertices = graph.get_vertices()
    for vertex in vertices:
        if vertex not in visited:
            # Nuevo componente encontrado, iniciamos BFS
            component = []
            queue = [vertex]
            visited.add(vertex)
            
            while queue:
                current = queue.pop(0)
                component.append(current)
                
                for neighbor in graph.adj_list.get(current, {}):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            components.append(component)
            
    return components

def is_bipartite(graph, component):
    """
    Determina si una componente del grafo es bipartita.

    Un grafo bipartito permite dividir sus vértices en dos grupos, de forma que
    no existan aristas entre vértices del mismo grupo. Para verificarlo, se usa
    BFS asignando colores alternos a los vértices vecinos.

    Parámetros:
        graph (FlightGraph): Grafo completo.
        component (list): Lista de vértices que pertenecen a la componente evaluada.

    Retorna:
        bool: True si la componente es bipartita, False en caso contrario.
    """
    if not component:
        return True, []
        
    color_map = {}
    parent_map = {}
    
    # Podría haber componentes desconectados dentro del parámetro (aunque por diseño viene conexo), iteramos de manera segura.
    for start_node in component:
        if start_node not in color_map:
            color_map[start_node] = 0
            parent_map[start_node] = None
            queue = [start_node]
            
            while queue:
                current = queue.pop(0)
                current_color = color_map[current]
                
                for neighbor in graph.adj_list.get(current, {}):
                    if neighbor in component: # Solo consideramos nodos en la componente
                        if neighbor not in color_map:
                            # Asignamos el color opuesto (1 - color actual)
                            color_map[neighbor] = 1 - current_color
                            parent_map[neighbor] = current
                            queue.append(neighbor)
                        elif color_map[neighbor] == current_color:
                            # Mismo color en nodos adyacentes = no es bipartito, hemos encontrado un ciclo impar
                            path_u = [current]
                            curr = parent_map.get(current)
                            while curr is not None:
                                path_u.append(curr)
                                curr = parent_map.get(curr)
                                
                            path_v = [neighbor]
                            curr = parent_map.get(neighbor)
                            while curr is not None:
                                path_v.append(curr)
                                curr = parent_map.get(curr)
                                
                            path_u_set = set(path_u)
                            lca = None
                            for node in path_v:
                                if node in path_u_set:
                                    lca = node
                                    break
                                    
                            cycle = []
                            for node in path_u:
                                cycle.append(node)
                                if node == lca:
                                    break
                                    
                            v_to_lca = []
                            for node in path_v:
                                v_to_lca.append(node)
                                if node == lca:
                                    break
                            
                            if v_to_lca:
                                v_to_lca.pop() # remove lca
                                v_to_lca.reverse()
                                cycle.extend(v_to_lca)
                                
                            cycle.append(current) # Cerrar el ciclo
                            
                            return False, cycle
    return True, []

# --- Estructura Disjoint-Set / Union-Find para Kruskal ---
class DisjointSet:
    """
    Estructura Disjoint Set, también conocida como Union-Find.

    Se utiliza en el algoritmo de Kruskal para detectar ciclos de manera eficiente.
    Cada vértice pertenece inicialmente a su propio conjunto. Al agregar una
    arista válida, los conjuntos de sus extremos se unen.

    Atributos:
        parent (dict): Representante o padre de cada elemento.
        rank (dict): Rango aproximado de cada conjunto para optimizar las uniones.
    """
    def __init__(self, elements):
        """
        Inicializa la estructura con un conjunto independiente por cada elemento.

        Parámetros:
            elements (list): Elementos que formarán los conjuntos iniciales.
        """
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}
        
    def find(self, item):
        """
        Encuentra el representante principal del conjunto al que pertenece un elemento.

        Usa compresión de caminos para acelerar futuras búsquedas.

        Parámetros:
            item (str): Elemento consultado.

        Retorna:
            str: Representante del conjunto.
        """
        if self.parent[item] == item:
            return item
        # Path compression Optimization
        self.parent[item] = self.find(self.parent[item])
        return self.parent[item]
        
    def union(self, set1, set2):
        """
        Une los conjuntos a los que pertenecen dos elementos.

        Si ambos elementos ya pertenecen al mismo conjunto, la unión no se realiza
        porque eso indicaría que la arista formaría un ciclo en Kruskal.

        Parámetros:
            set1 (str): Primer elemento.
            set2 (str): Segundo elemento.

        Retorna:
            bool: True si se realizó la unión, False si ya estaban en el mismo conjunto.
        """
        root1 = self.find(set1)
        root2 = self.find(set2)
        
        if root1 != root2:
            # Union by Rank Optimization
            if self.rank[root1] > self.rank[root2]:
                self.parent[root2] = root1
            elif self.rank[root1] < self.rank[root2]:
                self.parent[root1] = root2
            else:
                self.parent[root2] = root1
                self.rank[root1] += 1
            return True
        return False

def kruskal_mst(graph, component):
    """
    Calcula el árbol de expansión mínima de una componente usando Kruskal.

    El algoritmo ordena las aristas por peso y las agrega una a una, siempre que
    no formen ciclos. Para detectar ciclos se usa Disjoint Set.

    Parámetros:
        graph (FlightGraph): Grafo completo.
        component (list): Lista de vértices de la componente evaluada.

    Retorna:
        tuple: (peso_total, aristas_mst)
            peso_total (float): Suma de los pesos del MST.
            aristas_mst (list): Lista de aristas seleccionadas en formato (u, v, peso).
    """
    component_set = set(component)
    edges = []
    
    # Extraer aristas válidas de esta componente
    visited_edges = set()
    for u in component:
        for v, weight in graph.adj_list.get(u, {}).items():
            if v in component_set:
                edge_pair = tuple(sorted([u, v]))
                if edge_pair not in visited_edges:
                    visited_edges.add(edge_pair)
                    edges.append((weight, u, v))
                    
    # Kruskal requiere ordenar aristas por peso
    edges.sort(key=lambda x: x[0])
    
    ds = DisjointSet(component)
    mst_weight = 0.0
    mst_edges = []
    
    for weight, u, v in edges:
        if ds.union(u, v):
            mst_weight += weight
            mst_edges.append((u, v, weight))
            
    return mst_weight, mst_edges

def dijkstra(graph, source):
    """
    Calcula los caminos mínimos desde un aeropuerto origen hacia todos los demás.

    Se implementa el algoritmo de Dijkstra usando una cola de prioridad con heapq.
    Este algoritmo funciona correctamente porque todas las distancias son positivas.

    Parámetros:
        graph (FlightGraph): Grafo sobre el cual se calcularán los caminos mínimos.
        source (str): Código del aeropuerto origen.

    Retorna:
        dict: Diccionario con dos llaves:
            distances: distancia mínima desde el origen hasta cada vértice.
            previous_nodes: predecesor de cada vértice para reconstruir caminos.
    """
    # Explicación de estructura de retorno:
    # distances[v] -> distancia mínima desde 'source' a 'v'
    # prev[v] -> predecesor de 'v' en el camino mínimo. Útil para reconstruir camino.
    
    distances = {vertex: float('inf') for vertex in graph.get_vertices()}
    prev = {vertex: None for vertex in graph.get_vertices()}
    distances[source] = 0.0
    
    # Cola de prioridad que almacena tuplas (distanciaacumulada, vertice)
    pq = [(0.0, source)]
    
    while pq:
        current_distance, current_vertex = heapq.heappop(pq)
        
        # Si sacamos un registro obsoleto de la cola (por Python heapq al no borrar) lo ignoramos:
        if current_distance > distances[current_vertex]:
            continue
            
        for neighbor, weight in graph.adj_list.get(current_vertex, {}).items():
            distance = current_distance + weight
            
            # Camino más costo-efectivo encontrado
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                prev[neighbor] = current_vertex
                heapq.heappush(pq, (distance, neighbor))
                
    return {'distances': distances, 'previous_nodes': prev}

def get_shortest_path(dijkstra_result, target):
    """
    Reconstruye el camino mínimo desde el origen hasta un destino.

    La función usa el diccionario previous_nodes generado por Dijkstra. Se empieza
    desde el destino y se retrocede por sus predecesores hasta llegar al origen.

    Parámetros:
        dijkstra_result (dict): Resultado retornado por la función dijkstra.
        target (str): Código del aeropuerto destino.

    Retorna:
        list: Lista de códigos de aeropuertos en orden desde origen hasta destino.
              Si no existe camino, retorna una lista vacía.
    """
    prev = dijkstra_result['previous_nodes']
    path = []
    current = target
    
    # Si su predecesor es None y no es el source de origen (cuya distancia seria 0), es inalcanzable
    if prev.get(current) is None and dijkstra_result['distances'].get(current, float('inf')) != 0:
        return []

    while current is not None:
        path.insert(0, current)
        current = prev.get(current)
        
    return path


def degree_centrality(graph, component=None):
    """
    Calcula la centralidad de grado de los vértices.

    La centralidad de grado mide la importancia de un nodo por su número de
    conexiones directas en el grafo o en una componente específica.

    Parámetros:
        graph (FlightGraph): Grafo completo.
        component (list | None): Lista de vértices a evaluar. Si es None, se usan todos.

    Retorna:
        dict: Mapa {vertice: centralidad de grado normalizada}.
    """
    vertices = component if component is not None else graph.get_vertices()
    n = len(vertices)
    max_degree = n - 1 if n > 1 else 1
    centrality = {}
    for node in vertices:
        degree = len(graph.adj_list.get(node, {}))
        centrality[node] = degree / max_degree if max_degree > 0 else 0.0
    return centrality


def closeness_centrality(graph, component=None):
    """
    Calcula la centralidad de cercanía para los vértices de una componente.

    La centralidad de cercanía mide qué tan cerca está un nodo del resto de la
    componente en términos de distancia acumulada mínima.

    Parámetros:
        graph (FlightGraph): Grafo completo.
        component (list | None): Lista de vértices a evaluar. Si es None, se usan todos.

    Retorna:
        dict: Mapa {vertice: centralidad de cercanía}.
    """
    vertices = component if component is not None else graph.get_vertices()
    n = len(vertices)
    centrality = {}
    if n == 0:
        return centrality

    for node in vertices:
        if n == 1:
            centrality[node] = 0.0
            continue

        dijkstra_result = dijkstra(graph, node)
        total_distance = 0.0
        reachable_count = 0
        for other in vertices:
            if other == node:
                continue
            dist = dijkstra_result['distances'].get(other, float('inf'))
            if dist != float('inf'):
                total_distance += dist
                reachable_count += 1

        if reachable_count == 0 or total_distance == 0:
            centrality[node] = 0.0
        else:
            normalizer = reachable_count / (n - 1)
            centrality[node] = (reachable_count / total_distance) * normalizer
    return centrality


def remove_node_and_analyze(graph, code):
    """
    Crea una copia del grafo sin un aeropuerto y analiza sus componentes.

    Parámetros:
        graph (FlightGraph): Grafo original.
        code (str): Código del aeropuerto a remover.

    Retorna:
        tuple: (FlightGraph, list) grafo modificado y lista de componentes.
    """
    if code not in graph.adj_list:
        return None, []

    graph_copy = graph.copy()
    graph_copy.remove_airport(code)
    components = find_components(graph_copy)
    return graph_copy, components
