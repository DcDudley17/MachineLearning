import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial import Voronoi, voronoi_plot_2d
import geopandas as gpd

us_states = gpd.read_file("https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json")

taxi = pd.read_csv("taxiSample.csv")[["pickup_longitude", "pickup_latitude"]]
taxi = taxi.to_numpy()
taxi = taxi[taxi[:, 0] > -75]
taxi = taxi[taxi[:, 1] > 39]


kmeans = KMeans(20)
kmeans.fit(taxi)
centers = kmeans.cluster_centers_


plt.style.use('dark_background')

fig, ax = plt.subplots()
us_states.plot(ax=ax, color='none', edgecolor='white')


vor = Voronoi(centers)
voronoi_plot_2d(vor, show_vertices=False, show_points=False, line_colors="white", line_width=2.0, ax=ax)
ax.set_xlim([-85, -65])
ax.set_ylim([38, 45])
plt.scatter(taxi[:, 0], taxi[:, 1], c=kmeans.labels_, s=4.0)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c="red", s=4.0)


plt.show()