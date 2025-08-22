import folium
import pandas as pd
import math
from shapely.geometry import Point
from shapely.affinity import scale, rotate
from folium import Map, GeoJson
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import folium
from folium.features import DivIcon
from branca.element import Template, MacroElement


# class HeatMap:


#     def __init__(self, coordinates = [54.7251, -101.8494], fireRate = 0.5, fireSize = 0):

# Fire center location
# center_lat, center_lon = coordinates[0], coordinates[1]
center_lat, center_lon = 54.7251, -101.8494

# Fire spread parameters
# spread_rate_m_per_min = fireRate # e.g., fire spreads 0.5 m each minute

spread_rate_km_per_hour = .75  # e.g., fire spreads 0.5 km each hour
total_hours = 6  # visualize fire spread over 6 hours

# Colors and corresponding labels for legend
colors = ["#8B0000", "#B22222", "#DC143C", "#FF6347", "#FF8C00", "#FFD700"]
labels = ["Hour 1", "Hour 2", "Hour 3", "Hour 4", "Hour 5", "Hour 6"]

opacities = [0.7, 0.6, 0.5, 0.4, 0.3, 0.2]

# Create base map with satellite and labels
m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles=None)

# Starting corner for legend placement (adjust as needed)
legend_base_lat = center_lat - 0.07
legend_base_lon = center_lon - 0.08

# Place each color circle and label
for i, (color, label) in enumerate(zip(colors, labels)):
    offset = i * 0.004

    # Color dot
    folium.CircleMarker(
        location=[legend_base_lat + offset, legend_base_lon],
        radius=6,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=1,
        tooltip=label
    ).add_to(m)

    # Text label
    folium.Marker(
        location=[legend_base_lat + offset, legend_base_lon + 0.005],
        icon=DivIcon(
            icon_size=(100, 24),
            icon_anchor=(0, 0),
            html=f'<div style="font-size: 12px; color: white; font-family: Arial;">{label}</div>',
        )
    ).add_to(m)


folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles © Esri",
    name="Esri Satellite",
    overlay=False,
    control=True
).add_to(m)

folium.TileLayer(
    tiles="https://services.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}",
    attr="Tiles © Esri",
    name="Esri Boundaries and Places",
    overlay=True,
    control=True
).add_to(m)

# Add concentric circles representing fire spread over time
for hour in range(total_hours, 0, -1):  # outer to inner for proper layering
    radius_km = spread_rate_km_per_hour * hour
    folium.Circle(
        location=[center_lat, center_lon],
        radius=radius_km * 1000,  # convert km to metersl
        color=colors[hour - 1],
        fill=True,
        fill_color=colors[hour - 1],
        fill_opacity=opacities[hour - 1],
        weight=2,
        tooltip=f"Fire spread at hour {hour}: radius {radius_km} km"
    ).add_to(m)

# Add black marker at center
folium.Marker(
    location=[center_lat, center_lon],
    icon=folium.Icon(color="black", icon="fire"),
    tooltip="Fire Origin"
).add_to(m)



folium.LayerControl().add_to(m)

m.save("fire_spread_concentric_circles.html")
print("Map saved as fire_spread_concentric_circles.html")

