# =========================================================
# Seven Degrees on SNAP Facebook (AgentPy + NetworkX)
# =========================================================
import random
import agentpy as ap
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

# ---------- Config de entrada ----------
GRAPH_PATH = "facebook_combined.txt"   # ajusta si tu archivo está en otra ruta
ITERATIONS = 5                         # número de corridas con distinto origen
MAX_GRADOS = 7                         # hasta 7 grados de separación

# ---------- Cache global del grafo (evita recargar en cada corrida) ----------
_G_CACHE = None
def _load_graph_once(path):
    global _G_CACHE
    if _G_CACHE is None:
        # El archivo SNAP es lista de aristas: "u v" por línea
        _G_CACHE = nx.read_edgelist(path, create_using=nx.Graph(), nodetype=int)
    return _G_CACHE


# ---------- Modelo ----------
class SevenDegreesModel(ap.Model):
    """
    En cada corrida:
    - Carga (desde cache) el grafo real de Facebook (SNAP).
    - Elige un nodo origen al azar (o fijo si pasas p.source_id).
    - Hace un BFS por capas:
        * "frontera" = nodos a distancia d
        * "alcanzados" = todos los nodos a distancia <= d
    - Cada paso de simulación = avanzar 1 grado más en el BFS.
    """

    def setup(self):
        # Cargar grafo real (cacheado)
        self.G = _load_graph_once(GRAPH_PATH)

        # Elegir origen
        if hasattr(self.p, "source_id") and self.p.source_id is not None and self.p.source_id in self.G:
            self.source = self.p.source_id
        else:
            self.source = random.choice(list(self.G.nodes()))

        # Estructuras del BFS por capas
        self.alcanzados = set([self.source])
        self.frontera = set([self.source])

        # Registro inicial (grado 0: uno mismo)
        self.record('grado', 0)
        self.record('personas_conocidas', len(self.alcanzados))
        self.record('fraccion_conocida', len(self.alcanzados) / self.G.number_of_nodes())

    def step(self):
        # Construir la siguiente capa (siguiente grado)
        nueva_frontera = set()
        for u in self.frontera:
            # vecinos directos
            for v in self.G.neighbors(u):
                if v not in self.alcanzados:
                    nueva_frontera.add(v)

        # Actualizar conjuntos
        self.frontera = nueva_frontera
        self.alcanzados |= nueva_frontera

        # Registrar métricas de este grado (self.t = número de step actual)
        self.record('grado', self.t)  # 1, 2, ..., MAX_GRADOS
        self.record('personas_conocidas', len(self.alcanzados))
        self.record('fraccion_conocida', len(self.alcanzados) / self.G.number_of_nodes())

        # Parada temprana si ya no hay más nodos nuevos
        if not self.frontera:
            self.stop()


# ---------- Experimento ----------
parameter_ranges = {
    'steps': MAX_GRADOS,  # número de grados a explorar
    # 'source_id': 123,   # opcional: fija un origen concreto existente en el grafo
}

sample = ap.Sample(parameter_ranges)

# Varias corridas para ver variabilidad por origen/semilla
exp = ap.Experiment(SevenDegreesModel, sample, iterations=ITERATIONS, record=True)
results = exp.run()

# ---------- Gráfica con unidades más claras ----------
sns.set_theme()
df = results.arrange_variables()

# Convertir fracción a porcentaje
df['porcentaje_conocida'] = df['fraccion_conocida'] * 100

ax = sns.lineplot(
    data=df,
    x='grado',
    y='porcentaje_conocida',
    errorbar='sd'
)

ax.set_xlabel('Grados de separación (número de saltos entre personas)')
ax.set_ylabel('Porcentaje de la red alcanzada (%)')
ax.set_title('Cobertura de la red de Facebook (SNAP) por grados de separación')

# Mostrar ticks del 0% al 100%
ax.set_ylim(0, 100)
ax.set_yticks(range(0, 101, 10))  # cada 10%

ax.set_xticks(range(0, df['grado'].max() + 1))

plt.show(block=True)

