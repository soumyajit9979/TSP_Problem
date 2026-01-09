import pandas as pd
import numpy as np
import plotly.graph_objects as go

from math import radians, sin, cos, sqrt, atan2
from sklearn.cluster import KMeans


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

    coords = points_df[["Latitude", "Longitude"]].to_numpy()
    num_agents = len(agents_df)

    return coords, num_agents


def haversine_distance(p1, p2):
    R = 6371

    lat1, lon1 = radians(p1[0]), radians(p1[1])
    lat2, lon2 = radians(p2[0]), radians(p2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


def compute_distance_matrix(coords):
    n = len(coords)
    dist = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            dist[i, j] = haversine_distance(coords[i], coords[j])

    return dist


def learn_agent_centers(coords, num_agents):
    kmeans = KMeans(n_clusters=num_agents, random_state=42, n_init=10)
    labels = kmeans.fit_predict(coords)
    centers = kmeans.cluster_centers_

    return labels, centers


def assign_points(labels, num_agents, coords, centers, dist_matrix):
    agent_points = {}
    agent_starts = {}

    for agent_id in range(num_agents):
        indices = np.where(labels == agent_id)[0]
        agent_points[agent_id] = indices
        
        # Find point closest to cluster center as starting point
        if len(indices) > 0:
            center = centers[agent_id]
            distances_to_center = [
                haversine_distance(coords[idx], center) 
                for idx in indices
            ]
            closest_idx = indices[np.argmin(distances_to_center)]
            agent_starts[agent_id] = closest_idx

    return agent_points, agent_starts


def nearest_neighbor_tsp(indices, dist_matrix, start_idx=None):
    """Nearest neighbor TSP with optional starting point."""
    if start_idx is not None and start_idx in indices:
        route = [start_idx]
        remaining = set(indices) - {start_idx}
    else:
        route = [indices[0]]
        remaining = set(indices[1:])

    while remaining:
        last = route[-1]
        next_city = min(remaining, key=lambda x: dist_matrix[last, x])
        route.append(next_city)
        remaining.remove(next_city)

    return route


def two_opt_optimize(route, dist_matrix, max_iterations=1000):
    """2-opt local search optimization."""
    improved = True
    iterations = 0
    
    while improved and iterations < max_iterations:
        improved = False
        iterations += 1
        
        for i in range(1, len(route) - 1):
            for j in range(i + 1, len(route)):
                # Calculate current distance
                current_dist = (
                    dist_matrix[route[i-1], route[i]] + 
                    dist_matrix[route[j], route[(j+1) % len(route)]]
                )
                
                # Calculate distance after 2-opt swap
                new_dist = (
                    dist_matrix[route[i-1], route[j]] + 
                    dist_matrix[route[i], route[(j+1) % len(route)]]
                )
                
                # If improvement found, apply 2-opt swap
                if new_dist < current_dist:
                    route[i:j+1] = reversed(route[i:j+1])
                    improved = True
                    break
            
            if improved:
                break
    
    return route


def solve_all_agents(agent_points, agent_starts, dist_matrix):
    """Solve TSP for each agent with 2-opt refinement."""
    routes = {}

    for agent_id, indices in agent_points.items():
        if len(indices) <= 1:
            routes[agent_id] = list(indices)
        else:
            # Use ML-determined starting point
            start_point = agent_starts.get(agent_id, indices[0])
            
            # Build initial route with nearest neighbor
            initial_route = nearest_neighbor_tsp(indices, dist_matrix, start_point)
            
            # Apply 2-opt optimization for improvement
            optimized_route = two_opt_optimize(initial_route, dist_matrix)
            routes[agent_id] = optimized_route

    return routes


def route_distance(route, dist_matrix):
    if len(route) <= 1:
        return 0

    total = 0
    for i in range(len(route) - 1):
        total += dist_matrix[route[i], route[i+1]]

    total += dist_matrix[route[-1], route[0]]
    return total


def plot_routes(coords, routes):
    fig = go.Figure()
    colors = generate_colors(len(routes))

    for agent_id, route in routes.items():
        if len(route) == 0:
            continue

        path = coords[route]
        path = np.vstack([path, path[0]])
        color = colors[agent_id % len(colors)]

        fig.add_trace(go.Scattergeo(
            lon=path[:, 1],
            lat=path[:, 0],
            mode='lines+markers',
            name=f"Agent {agent_id+1}",
            line=dict(color=color, width=2),
            marker=dict(size=6)
        ))

    fig.update_layout(
        title="TSP Routes (ML + 2-opt)",
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

    coords, num_agents = load_data(file_path)

    dist_matrix = compute_distance_matrix(coords)

    # Learn optimal cluster centers
    labels, centers = learn_agent_centers(coords, num_agents)

    # Use centers for assignment and routing
    agent_points, agent_starts = assign_points(labels, num_agents, coords, centers, dist_matrix)
    routes = solve_all_agents(agent_points, agent_starts, dist_matrix)

    # Evaluation
    for agent_id, route in routes.items():
        dist = route_distance(route, dist_matrix)
        print(f"Agent {agent_id+1}: {dist:.2f} km")

    plot_routes(coords, routes)

if __name__ == "__main__":
    main()
