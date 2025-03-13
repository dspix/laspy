
import geopandas as gpd
import folium

# Load shapefile
shapefile_path = "C:/Users/Will.Rust/OneDrive - Cranfield University/postdoc/Environment/Projects/RESTRECO/sweep_paper/MODIS_Selected_250m_Grid.shp"
gdf = gpd.read_file(shapefile_path)

import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt


# Google-like basemap with attribution
tiles = cimgt.GoogleTiles()
tiles.attribution = "© Google Maps"

for idx, row in gdf.iterrows():
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": ccrs.Mercator()})

    # Get centroid
    centroid = row.geometry.centroid
    x_center, y_center = centroid.x, centroid.y

    # Define plot extent
    buffer_size = 5000
    x_min, x_max = x_center - buffer_size, x_center + buffer_size
    y_min, y_max = y_center - buffer_size, y_center + buffer_size

    # Add basemap with the attribution fix
    ax.add_image(tiles, 14)  # Higher zoom for more detail

    # Plot polygon
    ax.set_extent([x_min, x_max, y_min, y_max], crs=ccrs.Mercator())
    gdf[gdf.index == idx].plot(ax=ax, facecolor="none", edgecolor="red", linewidth=2, transform=ccrs.Mercator())

    # Save image
    plt.savefig(f"polygon_{idx}.png", dpi=300, bbox_inches="tight")
    plt.close()

print("Static maps saved for each polygon!")
