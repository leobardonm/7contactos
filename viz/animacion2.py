# =========================================================
# Animación: “7 grados de separación” con personitas por nodo
# Requisitos: networkx, matplotlib, (opcional) pillow para guardar GIF
# =========================================================

import os
import math
import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

# ------------------ Parámetros editables ------------------
GRAPH_PATH       = "facebook_combined.txt"   # ruta al .txt de SNAP
SOURCE_ID        = None                      # ID válido para fijar origen; None = aleatorio
MAX_GRADOS       = 6                         # hasta qué grado animar (0..MAX_GRADOS)
MAX_NODOS_VISTA  = 2000                      # límite de nodos en el subgrafo que se dibuja
LAYOUT           = "spring"                  # "spring" | "kamada" | "fr"
SPREAD           = 2.0                       # >1 separa más los nodos (solo para spring)
SPRING_ITERS     = 150                       # iteraciones del layout spring
INTERVAL_MS      = 1000                      # milisegundos entre frames
SAVE_GIF         = False                     # guarda GIF si True
SAVE_MP4         = False                     # requiere ffmpeg
RANDOM_SEED      = 42                        # reproducibilidad del layout

# Modo de render
RENDER_MODE      = "image"                   # "image" | "emoji"
PERSON_ICON_PATH = "viz/persona.png"         # usado si RENDER_MODE="image"
ICON_BASE_ZOOM   = 0.035                     # tamaño base del ícono
# ----------------------------------------------------------

random.seed(RANDOM_SEED)

def cargar_grafo(path: str) -> nx.Graph:
    if not os.path.exists(path):
        raise FileNotFoundError(f"No encuentro el archivo: {path}")
    return nx.read_edgelist(path, create_using=nx.Graph(), nodetype=int)

def elegir_origen(G: nx.Graph, source_id=None) -> int:
    if source_id is not None and source_id in G:
        return source_id
    return random.choice(list(G.nodes()))

def subgrafo_por_grados(G: nx.Graph, origen: int, cutoff: int, max_nodos: int):
    dist = nx.single_source_shortest_path_length(G, origen, cutoff=cutoff)
    nodos_alcance = list(dist.keys())
    if len(nodos_alcance) > max_nodos:
        deg = G.degree(nodos_alcance)
        deg_dict = {u: d for u, d in deg}
        nodos_ordenados = sorted(
            nodos_alcance,
            key=lambda u: (dist[u], -deg_dict.get(u, 0))
        )[:max_nodos]
        nodos_alcance = nodos_ordenados
    H = G.subgraph(nodos_alcance).copy()
    dist_H = {u: d for u, d in dist.items() if u in H}
    return H, dist_H

def posiciones(H: nx.Graph, layout: str, seed: int = 42) -> dict:
    if layout == "kamada":
        return nx.kamada_kawai_layout(H)
    if layout == "fr":
        return nx.fruchterman_reingold_layout(H, seed=seed)
    # spring con "SPREAD" para más separación
    k = (SPREAD / max(math.sqrt(max(H.number_of_nodes(), 1)), 1.0))
    return nx.spring_layout(H, seed=seed, k=k, iterations=SPRING_ITERS, threshold=1e-4)

def construir_capas(dist: dict[int,int], max_grados: int):
    capas = [set() for _ in range(max_grados + 1)]
    for u, d in dist.items():
        if 0 <= d <= max_grados:
            capas[d].add(u)
    return capas

def draw_people_nodes(ax, pos, nodos, colores_halo, tamanios_rel, modo, icon_img=None):
    """
    Dibuja nodos como personitas (emoji o imagen) con un halo de color debajo.
    - colores_halo: color del halo por nodo (string hex)
    - tamanios_rel: tamaño relativo (número pequeño) -> escala halo e ícono
    """
    # --- HALO por debajo (zorder bajo) ---
    xs = [pos[u][0] for u in nodos]
    ys = [pos[u][1] for u in nodos]
    halo_sizes = [ts * 13 for ts in tamanios_rel]  # factor empírico
    ax.scatter(xs, ys, s=halo_sizes, c=colores_halo, alpha=0.20, linewidths=0, zorder=1)

    # --- PERSONITA arriba (zorder alto) ---
    artists = []
    if modo == "emoji":
        for (u, ts) in zip(nodos, tamanios_rel):
            x, y = pos[u]
            fontsize = 6 + int(ts * 0.9)
            t = ax.text(x, y, "👤", fontsize=fontsize, ha='center', va='center', zorder=4)
            artists.append(t)
    elif modo == "image" and icon_img is not None:
        mx = max(tamanios_rel) if tamanios_rel else 1.0
        for (u, ts) in zip(nodos, tamanios_rel):
            x, y = pos[u]
            # zoom proporcional al tamaño relativo
            zoom = ICON_BASE_ZOOM * (0.7 + 0.3 * (ts / max(mx, 1e-6)))
            im = OffsetImage(icon_img, zoom=zoom)
            ab = AnnotationBbox(im, (x, y), frameon=False, zorder=4)
            ax.add_artist(ab)
            artists.append(ab)
    return artists

def animar(G: nx.Graph):
    origen = elegir_origen(G, SOURCE_ID)
    H, dist = subgrafo_por_grados(G, origen, cutoff=MAX_GRADOS, max_nodos=MAX_NODOS_VISTA)
    pos = posiciones(H, LAYOUT, seed=RANDOM_SEED)
    capas = construir_capas(dist, MAX_GRADOS)

    COLOR_DESCUBIERTO = "#2166f3"  # halo azul
    COLOR_FRONTERA    = "#ff8f00"  # halo naranja
    COLOR_OCULTO      = "#cfcfcf"  # halo gris clarito

    fig, ax = plt.subplots(figsize=(11, 8))
    plt.axis("off")

    nodos_todos = list(H.nodes())
    edges_todos = list(H.edges())
    total_subgrafo = H.number_of_nodes()

    # aristas debajo de todo
    def draw_edges(edges_visibles):
        nx.draw_networkx_edges(H, pos, edgelist=edges_visibles, ax=ax,
                               width=0.35, alpha=0.28, edge_color="#666666", zorder=0)

    # Cargar ícono si aplica
    icon_img = None
    if RENDER_MODE == "image":
        if os.path.exists(PERSON_ICON_PATH):
            try:
                icon_img = plt.imread(PERSON_ICON_PATH)
            except Exception as e:
                print("⚠️ No se pudo leer el ícono, se usará emoji. Error:", e)

    def dibujar_frame(t):
        ax.clear()
        ax.set_title(
            f"Grado {t} — Origen: {origen} — Nodos mostrados: {H.number_of_nodes()} (subgrafo)",
            fontsize=12
        )
        ax.axis("off")

        descubiertos = set().union(*capas[:t+1])
        frontera = capas[t] if 0 <= t < len(capas) else set()
        edges_visibles = [(u, v) for (u, v) in edges_todos if u in descubiertos and v in descubiertos]
        draw_edges(edges_visibles)

        # Colores y tamaños relativos
        colores = []
        tamanios = []
        for u in nodos_todos:
            if u in frontera:
                colores.append(COLOR_FRONTERA)
                tamanios.append(18)     # más grande para frontera
            elif u in descubiertos:
                colores.append(COLOR_DESCUBIERTO)
                tamanios.append(11)
            else:
                colores.append(COLOR_OCULTO)
                tamanios.append(4)

        # Personitas arriba de todo
        draw_people_nodes(ax, pos, nodos_todos, colores, tamanios, RENDER_MODE, icon_img)

        # Texto informativo
        alcanzados = len(descubiertos)
        porcentaje = 100.0 * alcanzados / total_subgrafo
        ax.text(
            0.02, 0.02,
            f"Alcanzados ≤ grado {t}: {alcanzados:,} nodos ({porcentaje:0.1f}% del subgrafo)\n"
            f"Frontera (grado {t}): {len(frontera):,} nodos",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
        )

        # Leyenda (solo halos — el ícono toma el color del halo)
        leg_ax = ax
        leg_ax.scatter([], [], s=18*13, c=COLOR_FRONTERA, alpha=0.20, label="Frontera (halo naranja)")
        leg_ax.scatter([], [], s=11*13, c=COLOR_DESCUBIERTO, alpha=0.20, label="Descubiertos (halo azul)")
        leg_ax.scatter([], [], s= 4*13, c=COLOR_OCULTO,    alpha=0.20, label="No descubiertos")
        ax.legend(loc="upper right", frameon=True)

    anim = FuncAnimation(fig, dibujar_frame, frames=range(0, MAX_GRADOS + 1),
                         interval=INTERVAL_MS, blit=False, repeat=True)

    # Guardado opcional
    os.makedirs("viz", exist_ok=True)
    if SAVE_GIF:
        out = os.path.join("viz", "seven_degrees_people.gif")
        try:
            anim.save(out, writer="pillow", dpi=130)
            print(f"✅ GIF guardado en: {out}")
        except Exception as e:
            print("⚠️ No se pudo guardar GIF (instala pillow).", e)
    if SAVE_MP4:
        out = os.path.join("viz", "seven_degrees_people.mp4")
        try:
            anim.save(out, writer="ffmpeg", dpi=130)
            print(f"✅ MP4 guardado en: {out}")
        except Exception as e:
            print("⚠️ No se pudo guardar MP4 (instala ffmpeg).", e)

    plt.show()

def main():
    print("Cargando grafo...")
    G = cargar_grafo(GRAPH_PATH)
    print(f"Grafo: {G.number_of_nodes():,} nodos, {G.number_of_edges():,} aristas")
    animar(G)

if __name__ == "__main__":
    main()
