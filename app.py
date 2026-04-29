"""
app.py

Aplicación principal del laboratorio de grafos de rutas aéreas.

Este archivo construye la interfaz gráfica usando Streamlit y permite ejecutar
las funcionalidades solicitadas en el laboratorio:
- Verificar conexidad del grafo.
- Verificar si la componente más grande es bipartita.
- Calcular el árbol de expansión mínima con Kruskal.
- Consultar aeropuertos más lejanos usando Dijkstra.
- Visualizar en un mapa el camino mínimo entre dos aeropuertos.
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import math
import os

from core.graph import FlightGraph
from core.algorithms import find_components, is_bipartite, kruskal_mst, dijkstra, get_shortest_path

st.set_page_config(page_title="Rutas Aéreas - Estructura de Datos 2", layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def load_graph():
    """
    Carga el grafo desde el archivo CSV y lo mantiene en caché.

    Streamlit vuelve a ejecutar el script cada vez que el usuario interactúa con
    la interfaz. Por eso se usa cache_resource, evitando reconstruir el grafo en
    cada interacción.

    Retorna:
        FlightGraph: Grafo construido a partir de flights_final.csv.
    """
    g = FlightGraph()
    data_path = os.path.join("data", "flights_final.csv")
    if os.path.exists(data_path):
        g.load_from_csv(data_path)
    return g

def render_map(graph, path_nodes=None):
    """
    Renderiza un mapa con la ruta mínima entre dos aeropuertos.

    Si se recibe una lista de vértices en path_nodes, se dibujan marcadores para
    el origen, el destino y los vértices intermedios. Además, se traza una línea
    que representa el camino calculado.

    Parámetros:
        graph (FlightGraph): Grafo con la información de los aeropuertos.
        path_nodes (list | None): Lista de códigos de aeropuertos que forman el camino.

    Retorna:
        folium.Map: Mapa generado con Folium.
    """
    # Centrar en una posición neutral
    m = folium.Map(location=[20, 0], zoom_start=2)

    # Si hay un path, marcamos los aeropuertos del path y la linea entre ellos
    if path_nodes:
        coords = []
        for i, code in enumerate(path_nodes):
            info = graph.airports[code]
            lat, lon = info['lat'], info['lon']
            coords.append((lat, lon))
            
            # Agregar marcadores
            color = 'green' if i == 0 else ('red' if i == len(path_nodes)-1 else 'blue')
            label = f"Origen: {code}" if i == 0 else (f"Destino: {code}" if i == len(path_nodes)-1 else f"Parada {i}: {code}")
            folium.Marker(
                [lat, lon],
                popup=f"{label} - {info['name']}<br>{info['city']}, {info['country']}",
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(m)
            
        # Trazar linea
        folium.PolyLine(coords, color='red', weight=2.5, opacity=0.8).add_to(m)
    else:
        # Por rendimiento, no podemos pintar los miles de nodos a la vez sin lag,
        # así que damos la indicación.
        pass
        
    return m

def main():
    """
    Función principal de la aplicación.

    Construye la interfaz de usuario, carga el grafo y muestra las secciones
    correspondientes a cada requisito del laboratorio.
    """
    st.title("Sistema Experimental de Rutas Aéreas (Grafo)")
    st.write("Laboratorio de Análisis de Grafos de Transporte Aéreo.")

    graph = load_graph()
    
    if not graph.get_vertices():
        st.error("No se encontraron vértices generados. Asegúrate de que `flights_final.csv` esté en la carpeta `data/`.")
        st.stop()

    # Operaciones pesadas cacheadas on-demand
    @st.cache_data
    def get_components_data():
        """
        Calcula y almacena en caché las componentes conexas del grafo.

        Esta operación puede ser costosa porque recorre el grafo completo.
        Al cachearla, se evita recalcularla cada vez que cambia la interfaz.
        """
        return find_components(graph)
        
    sidebar_menu = st.sidebar.radio("Navegación / Requisitos:", [
        "1. Conexidad y Componentes",
        "2. Grafo Bipartito",
        "3. Árbol de Expansión Mínima",
        "4. Top 10 Rutas Más Largas (Caminos Mínimos)",
        "5. Trazar Camino Mínimo (Mapa)"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Estadísticas del Grafo:**\n\n- Número de Vértices: {len(graph.get_vertices())}\n- Número de Aristas: {len(graph.get_edges())}")
    
    # ---------------- REQUISITO 1 ----------------
    if "1. Conexidad" in sidebar_menu:
        st.header("1. Determinación de Conexidad del Grafo")
        with st.spinner("Calculando Búsqueda en Anchura (BFS)..."):
            components = get_components_data()
            
        is_connected = len(components) == 1
        
        if is_connected:
            st.success(f"El grafo generado **es conexo**. Consiste de un bloque gigante de {len(components[0])} vértices.")
        else:
            st.warning(f"El grafo generado **no es conexo**.")
            st.write(f"Se encontraron **{len(components)} componentes conexas** en total.")
            st.write("A continuación se muestra la cantidad de vértices por cada componente (mostrando las 15 más grandes):")
            
            # Ordenamos componentes por tamaño de mayor a menor para visualizar
            components_sorted = sorted(components, key=len, reverse=True)
            for i, comp in enumerate(components_sorted[:15]):
                st.markdown(f"- **Componente {i+1}:** {len(comp)} vértices.")

    # ---------------- REQUISITO 2 ----------------
    elif "2. Grafo Bipartito" in sidebar_menu:
        st.header("2. Verificación de Grafo Bipartito")
        components = get_components_data()
        
        # Filtrar la más grande
        largest_component = max(components, key=len)
        
        st.write(f"Al haber revisado las {len(components)} componentes, procederemos a evaluar **la componente más grande** que cuenta con {len(largest_component)} vértices.")
        
        with st.spinner("Ejecutando recorrido bicolor en BFS..."):
            bipartite = is_bipartite(graph, largest_component)
            
        if bipartite:
            st.success("La componente más grande generada **ES bipartita**.")
        else:
            st.error("La componente más grande generada **NO ES bipartita** (contiene ciclos de longitud impar).")

    # ---------------- REQUISITO 3 ----------------
    elif "3. Árbol" in sidebar_menu:
        st.header("3. Árbol de Expansión Mínima (MST)")
        components = get_components_data()
        st.write("Calculando MST utilizando el **Algoritmo de Kruskal** implementado matemáticamente con estructura *Disjoint-Set*.")
        
        # Calculamos solo los que no son componentes de 1 nodo (que pesan 0 y no tienen aristas)
        valid_components = [c for c in components if len(c) > 1]
        valid_components.sort(key=len, reverse=True)
        
        st.write(f"Evaluando pesos del MST en las {len(valid_components)} componentes multinodo.")
        
        if st.button("Generar Cálculos de Kruskal"):
            with st.spinner("Procesando Union-Find recursivo... (Esto puede tardar unos segundos)"):
                for i, comp in enumerate(valid_components[:10]): # Mostramos las primeras 10 si hay cientos de islas pequeñas
                    peso, mst_edges = kruskal_mst(graph, comp)
                    st.success(f"**Componente {i+1}** ({len(comp)} vértices, {len(mst_edges)} aristas conectadas): Peso total **{peso:,.2f} kilómetros**.")
                if len(valid_components) > 10:
                    st.info(f"... y {len(valid_components) - 10} componentes adicionales.")

    # ---------------- REQUISITO 4 ----------------
    elif "4. Top 10" in sidebar_menu:
        st.header("4. Analítica de Caminos Mínimos (Dijkstra)")
        
        airports_list = sorted(list(graph.get_vertices()))
        selected_code = st.selectbox("Seleccione / Inserte el código IATA del Aeropuerto Origen (Ej: BOG, PTY, JFK):", airports_list)
        
        if selected_code:
            info = graph.airports[selected_code]
            st.markdown(f"### Información del Vértice de Origen:\n"
                        f"- **Código:** {info['code']}\n"
                        f"- **Nombre:** {info['name']}\n"
                        f"- **Ciudad:** {info['city']}\n"
                        f"- **País:** {info['country']}\n"
                        f"- **Coordenadas:** {info['lat']}, {info['lon']}")
            
            if st.button("Obtener Top 10 Aeropuertos Más Alejados (Camino Mínimo Más Largo)"):
                with st.spinner("Calculando Dijkstra sobre todas las aristas de la componente..."):
                    res = dijkstra(graph, selected_code)
                    distances = res['distances']
                    
                    # Filtramos infinitos (nodos inalcanzables en otras componentes)
                    reachable = {k: v for k, v in distances.items() if v != float('inf') and k != selected_code}
                    
                    # Ordenamos descendente
                    sorted_furthest = sorted(reachable.items(), key=lambda item: item[1], reverse=True)[:10]
                    
                    st.write("#### Top 10 Vuelos Más Largos según camino mínimo en malla:")
                    for idx, (dest_code, dist) in enumerate(sorted_furthest):
                        d_info = graph.airports[dest_code]
                        st.markdown(f"{idx+1}. **{dest_code} - {d_info['name']}** ({d_info['city']}, {d_info['country']})\n"
                                    f"    - *Distancia Acumulada:* **{dist:,.2f} km**\n"
                                    f"    - Coord: {d_info['lat']}, {d_info['lon']}")

    # ---------------- REQUISITO 5 ----------------
    elif "5. Trazar Camino" in sidebar_menu:
        st.header("5. Explorador Geográfico (Visualizador de Caminos Mínimos)")
        
        airports_list = sorted(list(graph.get_vertices()))
        
        col1, col2 = st.columns(2)
        with col1:
            source = st.selectbox("Aeropuerto de Origen (Código):", airports_list, index=0)
        with col2:
            target = st.selectbox("Aeropuerto de Destino (Código):", airports_list, index=1 if len(airports_list) > 1 else 0)
            
        if st.button("Buscar Camino y Trazar en Mapa"):
            if source == target:
                st.warning("El origen y destino son iguales.")
            else:
                with st.spinner("Calculando camino óptimo..."):
                    res = dijkstra(graph, source)
                    path = get_shortest_path(res, target)
                    dist = res['distances'][target]
                    
                if not path:
                    st.error(f"No existe un camino posible entre {source} y {target}. Pertenecen a diferentes componentes.")
                else:
                    st.success(f"Camino Mínimo Encontrado! Compuesto por {len(path)} terminales aéreas. Distancia total recorrida: **{dist:,.2f} km**.")
                    
                    st.write("### Vértices Intermedios del Camino:")
                    for step_idx, node_code in enumerate(path):
                        ninfo = graph.airports[node_code]
                        st.write(f"{step_idx+1}. **{node_code}**: {ninfo['name']} ({ninfo['city']}, {ninfo['country']}) -> Lat: {ninfo['lat']}, Lon: {ninfo['lon']}")
                    
                    st.write("### Mapa Interactivo:")
                    # Generate Map
                    m = render_map(graph, path)
                    st_folium(m, width=800, height=500, returned_objects=[])

if __name__ == "__main__":
    main()
