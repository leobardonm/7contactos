# =========================================================
# Animación: “7 grados de separación” en red real de SNAP
# Autor: Leobardo + ChatGPT
# Requisitos: networkx, matplotlib
#   pip install networkx matplotlib
# =========================================================

import os
import math
import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ------------------ Parámetros editables ------------------
GRAPH_PATH      = os.path.join("facebook_combined.txt")  # ruta al .txt de SNAP
SOURCE_ID       = None      # pon un ID válido para fijar origen, p.ej. 107; o deja None para aleatorio
MAX_GRADOS      = 6         # hasta qué grado animar (0..MAX_GRADOS)
MAX_NODOS_VISTA = 2000      # límite de nodos en el subgrafo que se dibuja (rendimiento)
LAYOUT          = "spring"  # "spring" | "kamada" | "fr"
INTERVAL_MS     = 1200      # milisegundos entre frames
SAVE_GIF        = True      # guarda GIF en ./viz/seven_degrees.gif si True
SAVE_MP4        = False     # requiere ffmpeg
RANDOM_SEED     = 42        # para reproducibilidad del layout
# ----------------------------------------------------------

random.seed(RANDOM_SEED)

def cargar_grafo(path: str) -> nx.Graph:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No encuentro el archivo: {path}")
    G = nx.read_edgelist(path, create_using=nx.Graph(), nodetype=int)
    return G

def elegir_origen(G: nx.Graph, source_id=None) -> int:
    if source_id is not None and source_id in G:
        return source_id
    return random.choice(list(G.nodes()))

def subgrafo_por_grados(G: nx.Graph, origen: int, cutoff: int, max_nodos: int) -> tuple[nx.Graph, dict[int,int]]:
    """ Toma el ego-grafo hasta 'cutoff' grados desde el origen y lo recorta a 'max_nodos' si es necesario.
        Retorna subgrafo H y el dict distancias (solo nodos en H). """
    dist = nx.single_source_shortest_path_length(G, origen, cutoff=cutoff)
    nodos_alcance = list(dist.keys())

    # Si hay demasiados nodos para dibujar, recortamos priorizando:
    # 1) menor distancia primero (0,1,2,...),
    # 2) dentro de cada distancia, mayor grado primero (más informativos visualmente).
    if len(nodos_alcance) > max_nodos:
        deg = G.degree(nodos_alcance)
        deg_dict = {u: d for u, d in deg}
        nodos_ordenados = sorted(
            nodos_alcance,
            key=lambda u: (dist[u], -deg_dict.get(u, 0))
        )[:max_nodos]
        nodos_alcance = nodos_ordenados

    H = G.subgraph(nodos_alcance).copy()
    # Filtramos distancias a los nodos que realmente quedaron en H
    dist_H = {u: d for u, d in dist.items() if u in H}
    return H, dist_H

def posiciones(H: nx.Graph, layout: str, seed: int = 42) -> dict:
    if layout == "kamada":
        return nx.kamada_kawai_layout(H)
    if layout == "fr":
        return nx.fruchterman_reingold_layout(H, seed=seed)
    # por defecto: spring
    # Ajuste k ~ 1/sqrt(n) ayuda a espaciar en redes medianas
    k = 1.0 / math.sqrt(max(H.number_of_nodes(), 1))
    return nx.spring_layout(H, seed=seed, k=k)

def construir_capas(dist: dict[int,int], max_grados: int) -> list[set[int]]:
    """ Devuelve lista de conjuntos: capas[0] = {origen}, capas[1] = nodos a distancia 1, etc. """
    capas = [set() for _ in range(max_grados + 1)]
    for u, d in dist.items():
        if 0 <= d <= max_grados:
            capas[d].add(u)
    return capas

def animar(G: nx.Graph):
    origen = elegir_origen(G, SOURCE_ID)
    H, dist = subgrafo_por_grados(G, origen, cutoff=MAX_GRADOS, max_nodos=MAX_NODOS_VISTA)
    pos = posiciones(H, LAYOUT, seed=RANDOM_SEED)
    capas = construir_capas(dist, MAX_GRADOS)

    # Colores: descubiertos (azul), frontera (naranja), no descubiertos (gris claro)
    COLOR_DESCUBIERTO = "#2166f3"
    COLOR_FRONTERA    = "#ff8f00"
    COLOR_OCULTO      = "#d0d0d0"

    fig, ax = plt.subplots(figsize=(10, 8))
    plt.title("Crecimiento por grados de separación (SNAP Facebook)")
    plt.axis("off")

    # Para persistir los artistas entre frames
    nod_artist = None
    edge_artist = None
    txt_info = None

    # Pre-cálculo para suavidad de animación
    nodos_todos = list(H.nodes())
    edges_todos = list(H.edges())
    grados_total = H.number_of_nodes()

    def dibujar_frame(t):
        nonlocal nod_artist, edge_artist, txt_info
        ax.clear()
        ax.set_title(f"Grado {t} — Origen: {origen} — Nodos mostrados: {H.number_of_nodes()} (subgrafo)")

        # Nodos descubiertos hasta grado t, frontera exactamente grado t
        descubiertos = set().union(*capas[:t+1])
        frontera = capas[t] if 0 <= t < len(capas) else set()

        # Subconjunto de aristas cuyas dos puntas están descubiertas
        edges_visibles = [(u, v) for (u, v) in edges_todos if u in descubiertos and v in descubiertos]

        # Colorear nodos
        colores = []
        tama = []
        for u in nodos_todos:
            if u in frontera:
                colores.append(COLOR_FRONTERA)
                tama.append(70)
            elif u in descubiertos:
                colores.append(COLOR_DESCUBIERTO)
                tama.append(30)
            else:
                colores.append(COLOR_OCULTO)
                tama.append(5)

        # Dibujar aristas visibles
        edge_artist = nx.draw_networkx_edges(H, pos, edgelist=edges_visibles, ax=ax, width=0.5, alpha=0.6)
        # Dibujar nodos
        nod_artist = nx.draw_networkx_nodes(H, pos, nodelist=nodos_todos, node_color=colores, node_size=tama, ax=ax)

        # Texto informativo
        alcanzados = len(descubiertos)
        porcentaje = 100.0 * alcanzados / grados_total
        txt = (
            f"Alcanzados ≤ grado {t}: {alcanzados:,} nodos "
            f"({porcentaje:0.1f}% del subgrafo mostrado)\n"
            f"Frontera (grado {t}): {len(frontera):,} nodos"
        )
        txt_info = ax.text(0.02, 0.02, txt, transform=ax.transAxes, fontsize=10,
                           bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        # Leyenda simple
        ax.scatter([], [], s=70, c=COLOR_FRONTERA, label="Frontera (grado t)")
        ax.scatter([], [], s=30, c=COLOR_DESCUBIERTO, label="Descubiertos (≤ t)")
        ax.scatter([], [], s=30, c=COLOR_OCULTO, label="No descubiertos")
        ax.legend(loc="upper right", frameon=True)

        return (nod_artist, edge_artist, txt_info)

    anim = FuncAnimation(fig, dibujar_frame, frames=range(0, MAX_GRADOS + 1), interval=INTERVAL_MS, blit=False, repeat=True)

    # Guardado opcional
    os.makedirs("viz", exist_ok=True)
    if SAVE_GIF:
        gif_path = os.path.join("viz", "seven_degrees.gif")
        try:
            anim.save(gif_path, writer="pillow", dpi=130)
            print(f"✅ GIF guardado en: {gif_path}")
        except Exception as e:
            print("⚠️ No se pudo guardar GIF (instala pillow).", e)

    if SAVE_MP4:
        mp4_path = os.path.join("viz", "seven_degrees.mp4")
        try:
            anim.save(mp4_path, writer="ffmpeg", dpi=130)
            print(f"✅ MP4 guardado en: {mp4_path}")
        except Exception as e:
            print("⚠️ No se pudo guardar MP4 (instala ffmpeg).", e)

    plt.show()


def main():
    print("Cargando grafo... (esto puede tardar unos segundos)")
    G = cargar_grafo(GRAPH_PATH)
    print(f"Grafo cargado: {G.number_of_nodes():,} nodos, {G.number_of_edges():,} aristas")
    animar(G)

if __name__ == "__main__":
    main()
