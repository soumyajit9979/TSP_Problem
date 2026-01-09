import pandas as pd
import numpy as np
import plotly.graph_objects as go
from math import radians, sin, cos, sqrt, atan2


def generate_colors(n):
    """Generate n distinct colors using HSL color space."""
    colors = []
    for i in range(n):
        hue = i / n
        saturation = 0.7
        lightness = 0.5
        
        c = f"hsl({int(hue * 360)}, {int(saturation * 100)}%, {int(lightness * 100)}%)"
        colors.append(c)
    return colors


def load_data(file_path):
    points_df = pd.read_excel(file_path, sheet_name="Lat-Long")
    agents_df = pd.read_excel(file_path, sheet_name="TSP agents")

    coordinates = points_df[["Latitude", "Longitude"]].to_numpy()
    num_agents = len(agents_df)

    return coordinates, num_agents



def haversine_distance(point1, point2):
    R = 6371

    lat1, lon1 = radians(point1[0]), radians(point1[1])
    lat2, lon2 = radians(point2[0]), radians(point2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c



def compute_distance_matrix(coords):
    n = len(coords)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = haversine_distance(coords[i], coords[j])

    return dist_matrix



def partition_locations_geographically(coords, num_agents):
    latitudes = coords[:, 0]
    
    # Sort indices by latitude
    sorted_indices = np.argsort(latitudes)
    
    # Divide into roughly equal groups
    labels = np.zeros(len(coords), dtype=int)
    points_per_agent = len(coords) // num_agents
    
    for i, idx in enumerate(sorted_indices):
        agent_id = min(i // points_per_agent, num_agents - 1)
        labels[idx] = agent_id
    
    return labels



def nearest_neighbor_tsp(indices, dist_matrix):
    route = [indices[0]]
    remaining = set(indices[1:])

    while remaining:
        last = route[-1]
        next_city = min(remaining, key=lambda x: dist_matrix[last, x])
        route.append(next_city)
        remaining.remove(next_city)

    return route



def solve_multi_agent_tsp(coords, labels, dist_matrix, num_agents):
    agent_routes = {}

    for agent_id in range(num_agents):
        indices = np.where(labels == agent_id)[0]

        if len(indices) == 0:
            continue

        route = nearest_neighbor_tsp(indices, dist_matrix)
        agent_routes[agent_id] = route

    return agent_routes




def route_distance(route, dist_matrix):
    total = 0
    for i in range(len(route) - 1):
        total += dist_matrix[route[i], route[i + 1]]
    total += dist_matrix[route[-1], route[0]]

    return total



def plot_routes(coords, agent_routes):
    fig = go.Figure()
    colors = generate_colors(len(agent_routes))

    for agent_id, route in agent_routes.items():
        path = coords[route]
        path = np.vstack([path, path[0]])
        color = colors[agent_id % len(colors)]

        fig.add_trace(go.Scattergeo(
            lon=path[:, 1],
            lat=path[:, 0],
            mode='lines+markers',
            name=f"Agent {agent_id + 1}",
            line=dict(color=color, width=2),
            marker=dict(size=6)
        ))

    fig.update_layout(
        title="TSP Routes (Heuristic)",
        hovermode='closest',
        geo=dict(
            projection_type='mercator',
            showland=True,
            showocean=True
        )
    )
    fig.show()



def main():
    file_path = "TSP1.xlsm"

    # Load data
    coords, num_agents = load_data(file_path)

    # Compute distances
    dist_matrix = compute_distance_matrix(coords)

    # Assign locations to agents using geographic heuristic
    labels = partition_locations_geographically(coords, num_agents)

    # Solve TSP per agent
    agent_routes = solve_multi_agent_tsp(coords, labels, dist_matrix, num_agents)

    # Print route distances
    for agent_id, route in agent_routes.items():
        dist = route_distance(route, dist_matrix)
        print(f"Agent {agent_id + 1}: {dist:.2f} km")

    # Plot routes
    plot_routes(coords, agent_routes)



if __name__ == "__main__":
    main()


