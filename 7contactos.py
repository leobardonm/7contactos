import agentpy as ap
import networkx as nx

# Visualization
import seaborn as sns
import random as rand

class ButtonModel(ap.Model):

    def setup(self):

        # Create a graph with n agents
        self.gente_total = self.p.gente
        self.gente_conocida = 1

    def update(self):

        # Record size of the biggest cluster


        self.record('gente_conocida', self.gente_conocida)

        # Record threads to button ratio
        self.record('numero_de_concexiones', self.p.n)

    def step(self):

        # Create random edges based on parameters

        for _ in range(int(self.p.n)):
            gente_por_conocer=rand.randint(10,50)
            gente_por_conocer=self.gente_conocida*gente_por_conocer*(1-(self.gente_conocida/self.gente_total))



            self.gente_conocida=self.gente_conocida+int(gente_por_conocer)
# Define parameter ranges
parameter_ranges = {
    'steps':1,
    'gente': 8000000000,  # Speed of connections per step
    'n': ap.Values(1,2,3,4,5,6,7)  # Number of agents
}

# Create sample for different values of n
sample = ap.Sample(parameter_ranges)

# Keep dynamic variables
exp = ap.Experiment(ButtonModel, sample, iterations=1, record=True)

# Perform 75 separate simulations (3 parameter combinations * 25 repetitions)
results = exp.run()

# Plot averaged time-series for discrete parameter samples
sns.set_theme()
sns.lineplot(
    data=results.arrange_variables(),
    x='numero_de_concexiones',
    y='gente_conocida',


);