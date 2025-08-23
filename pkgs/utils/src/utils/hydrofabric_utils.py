"""Functions to handle hydrofabric data and operations."""

import logging
import sqlite3
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import GeometryCollection, Point
from shapely.ops import unary_union

from .io_utils import read_table
from .validation_utils import check_columns_dataframe

logger = logging.getLogger(__name__)


def find_gages_within_buffer(
    gage_file: Path,
    gage_id_col: str,
    buffer: float,
    gdf: gpd.GeoDataFrame = None,
    hydrofabric_file: Path = None,
    id: str = None,
) -> tuple[list, list, gpd.GeoSeries]:
    """Find gages within a buffer zone around the hydrofabric.

    Args:
        gage_file: path to the donor gage file containing gage_id, longitude, and latitude
        gage_id_col: column name for gage ID in the gage file
        buffer: buffer distance in kilometers around the hydrofabric
        gdf: optional, a GeoDataFrame of hydrofabric polygons (if provided, hydrofabric_file is ignored)
        hydrofabric_file: optional, path to the hydrofabric file (.gpkg or .shp format)
        id: optional, identifier for the hydrofabric


    Returns:
        gages: list of gages within the buffered hydrofabric
        distances: list of distances from each gage to the hydrofabric centroid
        gdf_buffered: GeoDataFrame representing the buffered hydrofabric geometry

    """
    # check if required columns are present in gage file
    required_columns = {"longitude", "latitude", gage_id_col}
    check_columns_dataframe(gage_file, required_columns)

    # read in lat/lon of all gages
    gages = read_table(gage_file)
    if gages.empty:
        raise ValueError(f"No gages found in the gage file: {gage_file}")

    # create a GeoDataFrame of gages with geometry as points
    gage_gdf = gpd.GeoDataFrame(
        gages,
        geometry=[Point(xy) for xy in zip(gages["longitude"], gages["latitude"])],
        crs="EPSG:4326",
    )

    # read the hydrofabric file
    if gdf is None:
        if hydrofabric_file is None:
            raise ValueError("Either gdf or hydrofabric_file must be provided.")
        if not hydrofabric_file.exists():
            raise FileNotFoundError(f"Hydrofabric file not found: {hydrofabric_file}")

        # read the hydrofabric file
        if hydrofabric_file.suffix.lower() == ".gpkg":
            # For GeoPackage files, read the 'divides' layer
            gdf = gpd.read_file(hydrofabric_file, layer="divides")
        elif hydrofabric_file.suffix.lower() == ".shp":
            # For Shapefile, read the file directly
            gdf = gpd.read_file(hydrofabric_file)
        else:
            raise ValueError(f"Unsupported hydrofabric file format: {hydrofabric_file.suffix}")

    # Project to meters for accurate distance calculations
    gage_gdf = gage_gdf.to_crs(epsg=3857)
    gdf = gdf.to_crs(epsg=3857)

    # remove invalid geometries
    gdf1 = gdf[gdf.is_valid]
    if gdf1.empty:  # If no valid geometries, create a copy with buffered geometry
        if id:
            logger.debug(
                f"No valid geometries found in the hydrofabric for ID: {id}. "
                f"Creating a buffered geometry as workaround."
            )
        else:
            logger.debug("No valid geometries found in the hydrofabric. Creating a buffered geometry.")
        gdf1 = gdf.copy()
        gdf1["geometry"] = gdf1.buffer(0)

    # dissolve all polygons into one before bufferring
    combined_geom = unary_union(gdf1.geometry)

    # Create a buffer around the VPU polygon
    gdf_buffered = combined_geom.buffer(buffer * 1000)

    # Find gages in the buffered VPU
    gages = gage_gdf[gage_gdf.geometry.within(gdf_buffered)][gage_id_col].tolist()

    # compute distances of the gages to the hydrofabric centroid
    gage_gdf = gage_gdf[gage_gdf[gage_id_col].isin(gages)]
    distances = gage_gdf.distance(combined_geom.centroid)

    # convert distances meters to kilometers and round to integer
    distances = distances / 1000
    distances = distances.round(0).astype(int)

    if not gages:
        logger.debug(f"No gages found for {id}. Please check the gage file and hydrofabric file.")

    return gages, distances, gdf_buffered


def area_weighted_average(
    gdf_fine: gpd.GeoDataFrame,
    gdf_coarse: gpd.GeoDataFrame,
    value_col: str,
    fine_id_col: str = "divide_id",
    crs_proj: str = "EPSG:5070",
) -> gpd.GeoDataFrame:
    """Map area-weighted average of `value_col` from coarse polygons to fine polygons.

    Args:
        gdf_fine: GeoDataFrame with finer polygons
        gdf_coarse: GeoDataFrame with coarser polygons and the value to be averaged
        value_col: Name of the column in gdf_coarse to average
        fine_id_col: Name of unique identifier column in gdf_fine (default is 'divide_id')
        crs_proj: Projected CRS (in meters) for accurate area computation (default is 'EPSG:5070')

    Returns:
        gdf_fine with a new column: `{value_col}_weighted`

    """
    # Reproject to projected CRS
    gdf_fine = gdf_fine.to_crs(crs_proj).copy()
    gdf_coarse = gdf_coarse.to_crs(crs_proj).copy()

    # Ensure unique ID on fine polygons
    if fine_id_col not in gdf_fine.columns:
        gdf_fine = gdf_fine.reset_index(drop=True)
        gdf_fine[fine_id_col] = gdf_fine.index

    # Ensure valid geometries before overlay
    gdf_fine["geometry"] = gdf_fine.geometry.buffer(0)
    gdf_coarse["geometry"] = gdf_coarse.geometry.buffer(0)

    # Overlay to get intersections
    intersected = gpd.overlay(
        gdf_fine[[fine_id_col, "geometry"]], gdf_coarse[["geometry", value_col]], how="intersection"
    )

    # Remove rows with unsupported GeometryCollection geometry
    intersected = intersected[~intersected.geometry.apply(lambda g: isinstance(g, GeometryCollection))]

    # Optionally explode multipart geometries
    intersected = intersected.explode(index_parts=True, ignore_index=True)

    # Compute area of intersection
    intersected["area"] = intersected.geometry.area

    # Weighted value = value * area
    intersected["weighted_value"] = intersected[value_col] * intersected["area"]

    # Sum weighted values and areas for each fine polygon
    area_sum = intersected.groupby(fine_id_col)["area"].sum()
    value_sum = intersected.groupby(fine_id_col)["weighted_value"].sum()

    # Compute area-weighted average
    weighted_avg = value_sum / area_sum

    # Assign back to fine GeoDataFrame
    new_col = f"{value_col}_weighted"
    gdf_fine[new_col] = gdf_fine[fine_id_col].map(weighted_avg)

    return gdf_fine


class DistanceStore:
    def __init__(self, db_path="distances.db"):
        import sqlite3

        self.conn = sqlite3.connect(db_path)
        self._create_tables()

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS donors (
                donor_id   INTEGER PRIMARY KEY,
                donor_name TEXT UNIQUE
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS receivers (
                receiver_id   INTEGER PRIMARY KEY,
                receiver_name TEXT UNIQUE
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS distances (
                receiver_id INTEGER,
                donor_id    INTEGER,
                distance    REAL,
                PRIMARY KEY (receiver_id, donor_id),
                FOREIGN KEY (receiver_id) REFERENCES receivers(receiver_id),
                FOREIGN KEY (donor_id)    REFERENCES donors(donor_id)
            )
        """)
        self.conn.commit()

    def _get_or_insert(self, table, name):
        cur = self.conn.cursor()
        cur.execute(f"INSERT OR IGNORE INTO {table} ({table[:-1]}_name) VALUES (?)", (name,))
        self.conn.commit()
        cur.execute(f"SELECT {table[:-1]}_id FROM {table} WHERE {table[:-1]}_name = ?", (name,))
        return cur.fetchone()[0]

    def compute_and_store(self, gdf_donors, gdf_receivers, chunk_size=5000, max_distance=None):
        """
        Compute donor-receiver distances (in km) and store only those <= max_distance if provided.
        Works for both lat/lon (EPSG:4326) and projected CRS.
        """
        donors = gdf_donors.copy()
        receivers = gdf_receivers.copy()

        # Ensure CRS is defined
        if donors.crs is None or receivers.crs is None:
            raise ValueError("CRS is missing in one of the GeoDataFrames.")
        if donors.crs != receivers.crs:
            receivers = receivers.to_crs(donors.crs)

        # Use centroids
        donors["centroid"] = donors.geometry.centroid
        receivers["centroid"] = receivers.geometry.centroid

        donor_ids = {row.divide_id: self._get_or_insert("donors", row.divide_id) for row in donors.itertuples()}
        receiver_ids = {
            row.divide_id: self._get_or_insert("receivers", row.divide_id) for row in receivers.itertuples()
        }

        cur = self.conn.cursor()

        is_latlon = donors.crs.to_string() == "EPSG:4326"

        donor_coords = np.array([[p.x, p.y] for p in donors.centroid])
        radius = 6371.0  # Earth radius in km (for lat/lon)

        for start in range(0, len(receivers), chunk_size):
            recv_chunk = receivers.iloc[start : start + chunk_size]
            recv_coords = np.array([[p.x, p.y] for p in recv_chunk.centroid])

            if is_latlon:
                # Haversine vectorized
                lat_a = np.radians(donor_coords[:, 1])[None, :]
                lon_a = np.radians(donor_coords[:, 0])[None, :]
                lat_b = np.radians(recv_coords[:, 1])[:, None]
                lon_b = np.radians(recv_coords[:, 0])[:, None]

                dlat = lat_b - lat_a
                dlon = lon_b - lon_a
                a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_b) * np.cos(lat_a) * np.sin(dlon / 2.0) ** 2
                c = 2 * np.arcsin(np.sqrt(a))
                dist_matrix = radius * c  # km
            else:
                # Projected CRS: Euclidean distance
                dx = donor_coords[:, 0][None, :] - recv_coords[:, 0][:, None]
                dy = donor_coords[:, 1][None, :] - recv_coords[:, 1][:, None]
                dist_matrix = np.sqrt(dx**2 + dy**2) / 1000.0  # km

            rows = []
            for r_idx, recv in enumerate(recv_chunk.itertuples()):
                recv_id = receiver_ids[recv.divide_id]
                for d_idx, donor in enumerate(donors.itertuples()):
                    distance = int(round(dist_matrix[r_idx, d_idx]))  # float(dist_matrix[r_idx, d_idx])
                    if (max_distance is None) or (distance <= max_distance):
                        rows.append((recv_id, donor_ids[donor.divide_id], distance))

            cur.executemany("INSERT OR REPLACE INTO distances VALUES (?, ?, ?)", rows)
            self.conn.commit()

    def get_distances(self, receiver_name, donor_names):
        """Retrieve distances for one receiver vs. a list of donors."""
        cur = self.conn.cursor()
        cur.execute("SELECT receiver_id FROM receivers WHERE receiver_name=?", (receiver_name,))
        recv = cur.fetchone()
        if recv is None:
            return {}
        recv_id = recv[0]

        placeholders = ",".join(["?"] * len(donor_names))
        cur.execute(
            f"""
            SELECT d.donor_name, dist.distance
            FROM distances dist
            JOIN donors d ON dist.donor_id = d.donor_id
            WHERE dist.receiver_id = ?
              AND d.donor_name IN ({placeholders})
            """,
            [recv_id, *donor_names],
        )
        return dict(cur.fetchall())
