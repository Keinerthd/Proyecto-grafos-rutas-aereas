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
from core.algorithms import (
    find_components,
    is_bipartite,
    kruskal_mst,
    dijkstra,
    get_shortest_path,
    degree_centrality,
    closeness_centrality,
    remove_node_and_analyze,
)

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
        "5. Trazar Camino Mínimo (Mapa)",
        "6. Centralidad y Eliminación de Nodo"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.info(f"**Estadísticas del Grafo:**\n\n- Número de Vértices: {len(graph.get_vertices())}\n- Número de Aristas: {len(graph.get_edges())}")
    
    # ---------------- REQUISITO 1 ----------------
    if "1. Conexidad" in sidebar_menu:
        st.header("1. Determinación de Conexidad del Grafo")
        
        st.markdown("""
        > **Interpretación Analítica:** La red aérea mundial presenta una estructura altamente conectada donde la mayoría de aeropuertos pertenecen a una componente gigante. Sin embargo, existen pequeñas componentes aisladas que representan redes regionales o aeropuertos con baja conectividad.
        """)
        
        with st.spinner("Calculando Búsqueda en Anchura (BFS)..."):
            components = get_components_data()
            
        is_connected = len(components) == 1
        
        total_vertices = len(graph.get_vertices())
        table_data = []
        for i, comp in enumerate(sorted(components, key=len, reverse=True)[:15]):
            comp_set = set(comp)
            edges_count = 0
            visited_edges = set()
            for u in comp:
                for v in graph.adj_list.get(u, {}):
                    if v in comp_set:
                        edge = tuple(sorted([u, v]))
                        if edge not in visited_edges:
                            visited_edges.add(edge)
                            edges_count += 1
            
            pct = (len(comp) / total_vertices) * 100 if total_vertices > 0 else 0
            table_data.append({
                "Componente": i + 1,
                "Vértices": len(comp),
                "Aristas": edges_count,
                "% del Grafo": f"{pct:.2f}%"
            })
            
        st.subheader("Análisis Estructural de Componentes")
        st.table(table_data)
        
        if is_connected:
            st.success(f"El grafo generado **es conexo**. Consiste de un bloque gigante de {len(components[0])} vértices.")
        else:
            st.warning(f"El grafo generado **no es conexo**. Se encontraron **{len(components)} componentes conexas** en total.")
            
        st.subheader("Visualización de Componentes (Muestra)")
        st.write("A continuación se muestra una representación geográfica de un subconjunto de la componente principal y una componente aislada.")
        
        col1, col2 = st.columns(2)
        components_sorted = sorted(components, key=len, reverse=True)
        main_comp = components_sorted[0]
        small_comps = [c for c in components_sorted[1:] if len(c) > 1]
        small_comp = small_comps[0] if small_comps else (components_sorted[1] if len(components_sorted) > 1 else None)
        
        with col1:
            st.write("**Componente Principal (Muestra 100 nodos)**")
            m_main = folium.Map(location=[20, 0], zoom_start=1)
            sample_nodes = main_comp[:100] if len(main_comp) > 100 else main_comp
            for node in sample_nodes:
                info = graph.airports[node]
                folium.CircleMarker(
                    location=[info['lat'], info['lon']],
                    radius=3, color='blue', fill=True, popup=info['name']
                ).add_to(m_main)
            st_folium(m_main, width=400, height=300, key="m_main", returned_objects=[])
            
        with col2:
            if small_comp:
                st.write(f"**Componente Menor ({len(small_comp)} nodos)**")
                info = graph.airports[small_comp[0]]
                m_small = folium.Map(location=[info['lat'], info['lon']], zoom_start=4)
                for node in small_comp:
                    ninfo = graph.airports[node]
                    folium.Marker(
                        location=[ninfo['lat'], ninfo['lon']],
                        popup=ninfo['name'], icon=folium.Icon(color='red')
                    ).add_to(m_small)
                st_folium(m_small, width=400, height=300, key="m_small", returned_objects=[])
            else:
                st.write("No hay componentes menores para mostrar.")

    # ---------------- REQUISITO 2 ----------------
    elif "2. Grafo Bipartito" in sidebar_menu:
        st.header("2. Verificación de Grafo Bipartito")
        
        st.info("""
        **Definición Teórica:** Un grafo bipartito es aquel cuyos vértices pueden dividirse en dos conjuntos disjuntos sin que existan conexiones entre vértices del mismo conjunto. Un grafo es bipartito sí y solo sí no contiene **ciclos de longitud impar**.
        """)
        
        components = get_components_data()
        
        # Filtrar la más grande
        largest_component = max(components, key=len)
        
        st.write(f"Al haber revisado las {len(components)} componentes, procederemos a evaluar **la componente más grande** que cuenta con {len(largest_component)} vértices.")
        
        with st.spinner("Ejecutando recorrido bicolor en BFS..."):
            bipartite, odd_cycle = is_bipartite(graph, largest_component)
            
        if bipartite:
            st.success("La componente más grande generada **ES bipartita**.")
        else:
            st.error("La componente más grande generada **NO ES bipartita** (contiene ciclos de longitud impar).")
            st.markdown("""
            > **Interpretación Aplicada:** La presencia de ciclos impares evidencia una estructura compleja de interconexiones dentro de la red aérea, característica común en sistemas de transporte reales.
            """)
            
            st.subheader("Evidencia: Ciclo Impar Encontrado")
            st.write(f"Se detectó un ciclo de {len(odd_cycle)-1} aristas: " + " -> ".join(odd_cycle))
            
            m_cycle = folium.Map(location=[20, 0], zoom_start=2)
            coords = []
            for i, node in enumerate(odd_cycle):
                info = graph.airports[node]
                lat, lon = info['lat'], info['lon']
                coords.append((lat, lon))
                folium.Marker(
                    [lat, lon],
                    popup=f"Paso {i}: {node} - {info['name']}",
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m_cycle)
                
            folium.PolyLine(coords, color='red', weight=3, opacity=0.8).add_to(m_cycle)
            if coords:
                m_cycle.fit_bounds(m_cycle.get_bounds())
                
            st_folium(m_cycle, width=800, height=400, returned_objects=[])

    # ---------------- REQUISITO 3 ----------------
    elif "3. Árbol" in sidebar_menu:
        st.header("3. Árbol de Expansión Mínima (MST)")
        st.markdown("""
        > **Interpretación Analítica:** El árbol de expansión mínima logra mantener la conectividad de la red utilizando el menor número posible de conexiones, minimizando además el peso total de la red (distancia).
        """)
        
        components = get_components_data()
        st.write("Calculando MST utilizando el **Algoritmo de Kruskal** implementado matemáticamente con estructura *Disjoint-Set*.")
        
        # Calculamos solo los que no son componentes de 1 nodo
        valid_components = [c for c in components if len(c) > 1]
        valid_components.sort(key=len, reverse=True)
        
        if not valid_components:
            st.warning("No hay componentes aptas.")
        else:
            # Seleccionar componente para graficar (tamaño moderado)
            target_comp = next((c for c in valid_components if 10 <= len(c) <= 100), valid_components[0])
            peso, mst_edges = kruskal_mst(graph, target_comp)
            
            comp_set = set(target_comp)
            orig_edges_count = 0
            orig_edges = []
            visited_edges = set()
            for u in target_comp:
                for v in graph.adj_list.get(u, {}):
                    if v in comp_set:
                        edge = tuple(sorted([u, v]))
                        if edge not in visited_edges:
                            visited_edges.add(edge)
                            orig_edges_count += 1
                            orig_edges.append((u, v))
                            
            st.subheader("Comparación Matemática")
            st.table([{
                "Estructura": "Grafo Original (Subcomponente)",
                "Vértices": len(target_comp),
                "Aristas": orig_edges_count
            }, {
                "Estructura": "Árbol de Expansión Mínima (MST)",
                "Vértices": len(target_comp),
                "Aristas": len(mst_edges)
            }])
            
            st.subheader("Visualización del MST vs Grafo Original")
            col1, col2 = st.columns(2)
            first_node_info = graph.airports[target_comp[0]]
            
            with col1:
                st.write("**Grafo Original**")
                m_orig = folium.Map(location=[first_node_info['lat'], first_node_info['lon']], zoom_start=4)
                for u, v in orig_edges:
                    iu, iv = graph.airports[u], graph.airports[v]
                    folium.PolyLine([(iu['lat'], iu['lon']), (iv['lat'], iv['lon'])], color='gray', weight=1, opacity=0.5).add_to(m_orig)
                for node in target_comp:
                    info = graph.airports[node]
                    folium.CircleMarker([info['lat'], info['lon']], radius=3, color='blue', fill=True).add_to(m_orig)
                st_folium(m_orig, width=400, height=300, key="map_orig", returned_objects=[])
                
            with col2:
                st.write("**MST Generado**")
                m_mst = folium.Map(location=[first_node_info['lat'], first_node_info['lon']], zoom_start=4)
                for u, v, w in mst_edges:
                    iu, iv = graph.airports[u], graph.airports[v]
                    folium.PolyLine([(iu['lat'], iu['lon']), (iv['lat'], iv['lon'])], color='green', weight=2, opacity=0.8).add_to(m_mst)
                for node in target_comp:
                    info = graph.airports[node]
                    folium.CircleMarker([info['lat'], info['lon']], radius=3, color='green', fill=True).add_to(m_mst)
                st_folium(m_mst, width=400, height=300, key="map_mst", returned_objects=[])

            st.write("---")
            if st.button("Generar Cálculos de Kruskal para Componentes Mayores"):
                with st.spinner("Procesando Union-Find recursivo... (Esto puede tardar unos segundos)"):
                    for i, comp in enumerate(valid_components[:10]): 
                        peso, mst_edges = kruskal_mst(graph, comp)
                        st.success(f"**Componente {i+1}** ({len(comp)} vértices, {len(mst_edges)} aristas conectadas): Peso total **{peso:,.2f} kilómetros**.")
                    if len(valid_components) > 10:
                        st.info(f"... y {len(valid_components) - 10} componentes adicionales.")

    # ---------------- REQUISITO 4 ----------------
    elif "4. Top 10" in sidebar_menu:
        st.header("4. Analítica de Caminos Mínimos (Dijkstra)")
        
        st.markdown("""
        > **Interpretación Geográfica:** Las rutas más largas corresponden principalmente a regiones geográficamente aisladas o periféricas de la red mundial.
        """)
        
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
                    
                    m_top10 = folium.Map(location=[info['lat'], info['lon']], zoom_start=2)
                    folium.Marker([info['lat'], info['lon']], popup=f"Origen: {selected_code}", icon=folium.Icon(color='green')).add_to(m_top10)
                    
                    for idx, (dest_code, dist) in enumerate(sorted_furthest):
                        d_info = graph.airports[dest_code]
                        st.markdown(f"{idx+1}. **{dest_code} - {d_info['name']}** ({d_info['city']}, {d_info['country']})\n"
                                    f"    - *Distancia Acumulada:* **{dist:,.2f} km**\n"
                                    f"    - Coord: {d_info['lat']}, {d_info['lon']}")
                                    
                        path = get_shortest_path(res, dest_code)
                        coords = [(graph.airports[n]['lat'], graph.airports[n]['lon']) for n in path]
                        
                        folium.Marker([d_info['lat'], d_info['lon']], popup=f"Top {idx+1}: {dest_code} ({dist:,.0f} km)", icon=folium.Icon(color='red')).add_to(m_top10)
                        folium.PolyLine(coords, color='red', weight=1.5, opacity=0.6).add_to(m_top10)
                        
                    st.write("### Visualización Geográfica de Destinos Lejanos")
                    st_folium(m_top10, width=800, height=450, returned_objects=[])

    # ---------------- REQUISITO 5 ----------------
    elif "5. Trazar Camino" in sidebar_menu:
        st.header("5. Explorador Geográfico (Visualizador de Caminos Mínimos)")
        
        st.info("""
        **Base Matemática y Teórica:**
        - **Cálculo de Pesos (Haversine):** El peso de cada arista en el grafo representa la distancia geográfica real. Se utiliza la **Fórmula de Haversine**, que calcula la distancia circular mínima entre dos puntos sobre una esfera (Tierra) a partir de sus latitudes y longitudes.
        - **Algoritmo de Dijkstra:** El algoritmo de Dijkstra permite encontrar el camino de menor costo acumulado entre un nodo origen y un nodo destino dentro de un grafo ponderado.
        """)
        
        
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

    elif "6. Centralidad" in sidebar_menu:
        st.header("6. Centralidad y Eliminación de Nodo")
        st.markdown(
            """
            Este módulo calcula medidas de centralidad para la componente principal del grafo y simula qué ocurre cuando se elimina un aeropuerto del sistema.

            - **Centralidad de Grado:** mide la importancia de un aeropuerto por la cantidad de conexiones directas.
            - **Centralidad de Cercanía:** mide qué tan rápido se puede alcanzar el resto de la componente desde un aeropuerto.
            - **Eliminación de Nodo:** permite ver cómo cambia la cantidad de componentes y si el grafo sigue siendo conexo.
            """
        )

        components = get_components_data()
        largest_component = max(components, key=len)
        st.write(f"La componente más grande tiene **{len(largest_component)} aeropuertos**.")

        with st.spinner("Calculando centralidad... Esto puede tardar unos segundos..."):
            degree_cent = degree_centrality(graph, largest_component)
            closeness_cent = closeness_centrality(graph, largest_component)

        top_degree = sorted(degree_cent.items(), key=lambda item: item[1], reverse=True)[:10]
        top_closeness = sorted(closeness_cent.items(), key=lambda item: item[1], reverse=True)[:10]

        st.subheader("Top 10 Aeropuertos según Centralidad de Grado")
        degree_rows = []
        for code, value in top_degree:
            info = graph.airports[code]
            degree_rows.append({
                "Código": code,
                "Aeropuerto": info['name'],
                "Ciudad": info['city'],
                "País": info['country'],
                "Centralidad de Grado": f"{value:.4f}",
                "Conexiones": len(graph.adj_list.get(code, {}))
            })
        st.table(degree_rows)

        st.subheader("Top 10 Aeropuertos según Centralidad de Cercanía")
        closeness_rows = []
        for code, value in top_closeness:
            info = graph.airports[code]
            closeness_rows.append({
                "Código": code,
                "Aeropuerto": info['name'],
                "Ciudad": info['city'],
                "País": info['country'],
                "Centralidad de Cercanía": f"{value:.6f}"
            })
        st.table(closeness_rows)

        st.write("---")
        st.subheader("Simulación de Eliminación de Nodo")
        airports_list = sorted(graph.get_vertices())
        removed_node = st.selectbox("Seleccione el aeropuerto que desea quitar del sistema:", airports_list)

        if st.button("Simular eliminación del aeropuerto"):
            with st.spinner("Analizando la estructura tras eliminar el nodo..."):
                removed_graph, new_components = remove_node_and_analyze(graph, removed_node)

            if removed_graph is None:
                st.error(f"El aeropuerto {removed_node} no existe en el grafo.")
            else:
                original_is_connected = len(components) == 1
                new_is_connected = len(new_components) == 1
                largest_after = max(len(c) for c in new_components) if new_components else 0
                sizes = sorted([len(c) for c in new_components], reverse=True)

                st.write(f"- Aeropuertos restantes: **{len(removed_graph.get_vertices())}**")
                st.write(f"- Componentes después de eliminar {removed_node}: **{len(new_components)}**")
                st.write(f"- Tamaño de la componente más grande después de la eliminación: **{largest_after}**")

                if original_is_connected:
                    if new_is_connected:
                        st.success("El grafo sigue siendo conexo tras eliminar el aeropuerto seleccionado.")
                    else:
                        st.warning("El grafo ya no es conexo tras eliminar este aeropuerto.")
                        st.write("Esto indica que el aeropuerto actúa como un punto de articulación en la red.")
                else:
                    st.info("El grafo original no era conexo. Se analiza el efecto global de la eliminación en la estructura de componentes.")
                    if new_is_connected:
                        st.success("Tras la eliminación, el grafo resultante quedó conexo.")
                    else:
                        st.warning("Tras la eliminación, el grafo resultante sigue sin ser conexo.")

                component_table = []
                for i, size in enumerate(sizes[:5], start=1):
                    component_table.append({
                        "Componente": i,
                        "Tamaño": size
                    })
                st.table(component_table)

                if len(sizes) > 5:
                    st.write(f"...y {len(sizes) - 5} componentes adicionales más pequeños.")

                st.markdown(
                    """
                    > **Interpretación práctica:** Si la eliminación de un nodo aumenta el número de componentes o reduce fuertemente el tamaño de la componente principal, ese aeropuerto es crítico para la conectividad de la red.
                    """
                )

if __name__ == "__main__":
    main()
