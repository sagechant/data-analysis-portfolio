import subprocess
import sys

def install(package):
    """Install package using pip."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure networkx is installed
try:
    import networkx as nx
except ModuleNotFoundError:
    print("networkx not found. Installing networkx...")
    install("networkx")
    import networkx as nx

# Ensure plotly is installed
try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    print("plotly not found. Installing plotly...")
    install("plotly")
    import plotly.graph_objects as go

# Optional: Install pygraphviz for enhanced layouts
try:
    import pygraphviz
except ModuleNotFoundError:
    print("pygraphviz not found. For enhanced layouts, install it via 'pip install pygraphviz'")

# -----------------------------
# Begin the workflow visualization
# -----------------------------

# Create a directed graph
G = nx.DiGraph()

# Define nodes with labels and descriptions for hover text
nodes = {
    "A": "Data Integration & Preprocessing",
    "B1": "Cryo-ET Data Processing (3D atomic structures)",
    "B2": "Live Imaging Data Processing (Temporal dynamics)",
    "B3": "Super-Resolution Data Processing (Spatial mapping)",
    "C": "ML Alignment & Feature Extraction (CNNs, RNNs, GNNs)",
    "D": "Quantum Computing for Phosphorylation Analysis",
    "D1": "Quantum Chemistry Simulations (VQE, QPE)",
    "D2": "Quantum Machine Learning (QNNs, QAOA)",
    "E": "Hybrid Quantum-Classical Integration",
    "E1": "Quantum-Derived Parameters (Charge distribution, H-bonds)",
    "E2": "Classical MD Simulations (GROMACS, PINNs)",
    "F": "Validation & Iterative Refinement (ML classifiers, Bayesian optimization)",
    "G": "Predictive Modeling & Biological Insights (Phosphorylation hotspots, migration metrics)"
}

# Add nodes to the graph
for node, label in nodes.items():
    G.add_node(node, label=label)

# Define the edges based on the project workflow
edges = [
    ("A", "B1"), ("A", "B2"), ("A", "B3"),
    ("B1", "C"), ("B2", "C"), ("B3", "C"),
    ("C", "D"),
    ("D", "D1"), ("D", "D2"),
    ("D1", "E1"), ("D2", "E1"),
    ("E1", "E2"),
    ("E2", "F"),
    ("F", "G")
]

G.add_edges_from(edges)

# Try to use Graphviz layout; if unavailable, fallback to spring layout.
try:
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
except Exception as e:
    print("Graphviz layout not available. Falling back to spring layout.")
    pos = nx.spring_layout(G)

# Prepare edge coordinates for Plotly
edge_x = []
edge_y = []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.extend([x0, x1, None])
    edge_y.extend([-y0, -y1, None])  # Invert y for a top-to-bottom layout

edge_trace = go.Scatter(
    x=edge_x, 
    y=edge_y,
    line=dict(width=1.5, color='#888'),
    hoverinfo='none',
    mode='lines'
)

# Prepare node coordinates and hover text
node_x = []
node_y = []
node_text = []
for node in G.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(-y)  # Invert y for layout consistency
    node_text.append(f"{node}: {G.nodes[node]['label']}")

node_trace = go.Scatter(
    x=node_x, 
    y=node_y,
    mode='markers+text',
    text=[node for node in G.nodes()],
    textposition="bottom center",
    hoverinfo='text',
    marker=dict(
        color='LightSkyBlue',
        size=25,
        line_width=2
    ),
    hovertext=node_text
)

# Create the interactive Plotly figure with updated title settings
fig = go.Figure(
    data=[edge_trace, node_trace],
    layout=go.Layout(
        title={
            'text': '<br>Interactive Workflow: Vimentin Dynamics Simulation Project',
            'font': {'size': 20}
        },
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        annotations=[dict(
            text="This diagram integrates quantum computing, ML, and classical simulations for modeling vimentin dynamics.",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.1
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
    )
)

fig.show()
