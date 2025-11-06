"""Create NextGen catchment to attribute dataset subbasin crosswalk table."""

import argparse
import gc
from pathlib import Path

import fiona
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import box


def create_cwt(
    shp1: gpd.GeoDataFrame,
    shp2: gpd.GeoDataFrame,
    overlap_threshold_max: float = 99.0,
    overlap_threshold_min=20.0,
):
    """Create crosswalk table between two sets of polygons based on overlapping area percentage.

    For each polygon in shp2, find polygons in shp1 that overlap more than the specified threshold percentage.
    """
    # Convert to a projected CRS for accurate area calculations
    # projected_crs = "EPSG:3857"
    projected_crs = "EPSG:6933"
    shp1 = shp1.to_crs(projected_crs)
    shp2 = shp2.to_crs(projected_crs)

    # filter both shapefiles to only include polygons
    shp2 = shp2[shp2.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]
    shp1 = shp1[shp1.geometry.geom_type.isin(["Polygon", "MultiPolygon"])]

    # Find overlapping areas with shp1
    overlap = gpd.overlay(shp2, shp1, how="intersection", keep_geom_type=True)
    print(f"overlap: {overlap}")
    # overlap = overlap[overlap.geometry.geom_type.isin(["Polygon", "MultiPolygon"])] # caution: this filters out some valid overlaps

    # Compute area of each original polygon in shp2 (and convert to km^2)
    shp2["original_area"] = shp2.geometry.area / 1_000_000

    # Compute intersection area (and convert to km^2)
    overlap["overlap_area"] = overlap.geometry.area / 1_000_000
    overlap = overlap.drop(columns="areasqkm")

    # merge overlap with shp2 (to get original_area)
    overlap = overlap.merge(shp2.drop(columns="geometry"), on="divide_id", how="left")

    # Calculate percentage of each shp2 polygon that is covered by intersecting polygons in shp1
    overlap["overlap_percentage"] = (
        overlap["overlap_area"] / overlap["original_area"]
    ) * 100

    # sort by divide_id and descending overlap_percentage
    overlap = overlap.sort_values(
        by=["divide_id", "overlap_percentage"], ascending=[True, False]
    )

    # compute cumulative overlap_percentage for each divide
    overlap["cum_percentage"] = overlap.groupby("divide_id")[
        "overlap_percentage"
    ].cumsum()

    # filter out shp1 polygons when cumulative overlap_percentage reaches the max threshold requirement
    # (but always keep the first row)
    overlap["rank"] = overlap.groupby("divide_id").cumcount()
    overlap = overlap[
        (overlap["rank"] == 0) | (overlap["cum_percentage"] <= overlap_threshold_max)
    ]

    # make sure (maximum) cumulative overlap percentage is greater than the minimum threshold
    # Find ids where max(cum_area) > threshold
    polys_matched = overlap.groupby("divide_id")["cum_percentage"].max()
    polys_matched = polys_matched[polys_matched >= overlap_threshold_min].index
    overlap = overlap[overlap["divide_id"].isin(polys_matched)]

    # identify shp2 polygons not paired with a shp1 polygon
    polys = shp2["divide_id"].unique()
    polys_unmatched = [x for x in polys if x not in polys_matched]

    # drop columns that are no longer needed
    overlap = overlap.drop(
        columns=["overlap_area", "original_area", "cum_percentage", "rank"]
    )

    # Find the nearest shp1 polygon for each unmatched shp2 polygon
    nearest_matches = pd.DataFrame()
    if polys_unmatched:
        print(
            f"Number of polygons without sufficient overlap match (using nearest neighbor): {len(polys_unmatched)}"
        )
        nearest_matches = gpd.sjoin_nearest(
            shp2[shp2["divide_id"].isin(polys_unmatched)],
            shp1,
            how="left",
            distance_col="nearest_distance",
        )
        nearest_matches.rename(
            columns={"nearest_distance": "nearest_dist_m"}, inplace=True
        )
        nearest_matches = nearest_matches[["divide_id", "id", "nearest_dist_m"]]

    # create the final shp2/shp1 crosswalk table
    cwt = pd.concat(
        [overlap.drop(columns="geometry"), nearest_matches],
        axis=0,
        ignore_index=True,
    )

    return cwt


def get_attr_info(attr: str, vpu: str) -> dict:
    """Get attribute dataset information based on the attribute name."""
    attr_info = dict()
    match attr.lower():
        case "hlr":
            attr_info["name"] = "hlr"
            attr_info["file"] = Path("~/data/HLR/hlrshape/hlrus.shp").expanduser()
            attr_info["id_column"] = "VALUE"
            attr_info["layer"] = "hlrus"
        case "hydroatlas":
            attr_info["name"] = "hydroatlas"
            attr_info["file"] = Path(
                "~/data/HydroATLAS/BasinATLAS_v10_shp/BasinATLAS_v10_lev12.shp"
            ).expanduser()
            attr_info["id_column"] = "PFAF_ID"
            attr_info["layer"] = "BasinATLAS_v10_lev12"
        case "nhdplus" | "streamcat":
            attr_info["name"] = "streamcat"
            if vpu == "hi":
                vpu = "20"
            elif vpu == "prvi":
                vpu = "21"
            elif vpu == "ak":
                print("WARNING: NHDPlus/StreamCat data not available for AK.")
                return {}

            attr_info["file"] = Path(
                f"~/data/NHDPlusV21/NHD_by_vpu/NHDPlus{vpu}/NHDPlusCatchment/Catchment.shp"
            ).expanduser()
            attr_info["id_column"] = "FEATUREID"
            attr_info["layer"] = "Catchment"
        case _:
            raise Exception(f"Unsupported attribute dataset: {attr}")

    return attr_info


def get_vpu_list(domain: str) -> list:
    """Get list of VPUs for the specified domain."""
    match domain.lower():
        case "conus":
            vpu_list = [
                "01",
                "02",
                "03N",
                "03S",
                "03W",
                "04",
                "05",
                "06",
                "07",
                "08",
                "09",
                "10L",
                "10U",
                "11",
                "12",
                "13",
                "14",
                "15",
                "16",
                "17",
                "18",
            ]
        case "ak":
            vpu_list = ["ak"]
        case "hi":
            vpu_list = ["hi"]
        case "prvi":
            vpu_list = ["prvi"]
        case _:
            raise Exception(f"Unsupported domain: {domain}")

    return vpu_list


def get_vpu_gpd(vpu: str, shp_dir: str | Path) -> gpd.GeoDataFrame:
    """Get GeoDataFrame for the specified VPU."""
    vpu_files = list(Path(shp_dir).expanduser().glob("*.gpkg"))
    vpu_file = [f for f in vpu_files if f"vpu_{vpu}" in f.name]
    if not vpu_file:
        print(f"Warning: vpu file not found for vpu {vpu}. Skipping...")
        return gpd.GeoDataFrame()
    elif len(vpu_file) > 1:
        print(f"Warning: multiple vpu files found for vpu {vpu}. Skipping...")
        return gpd.GeoDataFrame()
    else:
        vpu_file = vpu_file[0]

    shp_vpu = gpd.read_file(vpu_file, layer="divides")
    shp_vpu = shp_vpu[["divide_id", "areasqkm", "geometry"]]
    return shp_vpu


def process_cwt_by_vpu(
    attr: str,
    cwt_file: Path,
    vpu: str,
    shp_dir: str,
    threshold_max: float = 99.0,
    threshold_min=30.0,
) -> pd.DataFrame:
    """Process crosswalk table creation for NextGen catchments for the specified VPU."""
    # get the attribute dataset info
    attr_dict = get_attr_info(attr, vpu)
    if not attr_dict:
        return pd.DataFrame()

    # read NextGen divides for the vpu
    shp_vpu = get_vpu_gpd(vpu, shp_dir)

    # read attribute dataset sub-basins given the vpu bounding box
    with fiona.open(attr_dict["file"]) as src:
        crs_attr = src.crs
    bbox = shp_vpu.total_bounds
    bbox_geom = gpd.GeoSeries([box(*bbox)], crs=shp_vpu.crs)
    bbox_reprojected = bbox_geom.to_crs(crs_attr)
    bbox_bounds = bbox_reprojected.total_bounds
    bbox_geom1 = box(*bbox_bounds)
    shp_attr = gpd.read_file(
        attr_dict["file"], layer=attr_dict["layer"], bbox=bbox_geom1
    )

    # rename id column for processing in create_cwt
    shp_attr.rename(columns={attr_dict["id_column"]: "id"}, inplace=True)
    shp_attr = shp_attr[["id", "geometry"]]

    if shp_attr.empty:
        print(f"Warning: no overlapping {attr} subbasins found for vpu {vpu}")
        return pd.DataFrame()

    # create crosswalk
    cwt1 = create_cwt(
        shp_attr,
        shp_vpu,
        overlap_threshold_max=threshold_max,
        overlap_threshold_min=threshold_min,
    )
    if cwt1.empty:
        print(f"Warning: no crosswalk created for vpu {vpu}")
        return pd.DataFrame()

    # reset id column back
    cwt1.rename(columns={"id": attr_dict["id_column"]}, inplace=True)

    # save cwt for the current vpu
    cwt_file.parent.mkdir(parents=True, exist_ok=True)
    cwt1.to_parquet(cwt_file, engine="pyarrow")

    del shp_attr, shp_vpu
    gc.collect()

    return cwt1


def process_cwt_domain(
    attr: str,
    domain: str,
    cwt_dir: Path,
    shp_dir: Path,
    threshold_max: float = 99.0,
    threshold_min=30.0,
) -> pd.DataFrame:
    """Process crosswalk table creation for NextGen catchments for the specified domain."""
    # get the list of vpus for the domain
    vpu_list = get_vpu_list(domain)

    # process cwt by vpu to reduce memory usage
    df_cwt = pd.DataFrame()
    for vpu in vpu_list:
        # first check if cwt file already exists
        cwt_file = Path(
            cwt_dir, "cwt_ngen_" + attr, "cwt_ngen_" + attr + "_vpu_" + vpu + ".parquet"
        )
        if cwt_file.exists():
            print(f"CWT file already exists for vpu {vpu}. Reading from  {cwt_file}")
            cwt1 = pd.read_parquet(cwt_file, engine="pyarrow")
        else:
            print(f"Processing crosswalk for vpu {vpu}...")
            cwt1 = process_cwt_by_vpu(
                attr,
                cwt_file,
                vpu,
                shp_dir,
                threshold_max,
                threshold_min,
            )
        # add to conus dataframe
        cwt1["vpuid"] = vpu
        df_cwt = pd.concat([df_cwt, cwt1])

        # free up memory
        del cwt1
        gc.collect()

    # save the combined cwt
    if df_cwt.empty:
        print(
            f"WARNING: No crosswalk table created for {attr} for the {domain} domain."
        )
    else:
        if domain.upper() == "CONUS":
            output_dir = Path(
                cwt_file.parent, "cwt_ngen_" + attr + "_conus.parquet"
            ).expanduser()
            print(f"Saving crosswalk table for {attr} to {output_dir}")
            df_cwt.to_parquet(
                output_dir,
                engine="pyarrow",
                index=False,
            )

        # analyze cwt results
        analyze_cwt_results(
            df_cwt,
            domain=domain,
            attr=attr,
            output_dir=cwt_file.parent,
            vpu_shp_dir=shp_dir,
        )

    return df_cwt


def analyze_cwt_results(
    df_cwt: pd.DataFrame,
    domain: str = "conus",
    attr: str = "hydroatlas",
    output_dir: Path = None,
    vpu_shp_dir: str | Path = None,
):
    """Analyze crosswalk table results and print summary statistics."""
    attr_dict = get_attr_info(attr, vpu="01")

    cats_total = df_cwt["divide_id"].unique()
    df_cwt_unmatched = df_cwt[~df_cwt["nearest_dist_m"].isna()]
    cats_unmatched = df_cwt_unmatched["divide_id"].unique().tolist()
    subs_unmatched = df_cwt_unmatched[attr_dict["id_column"]].unique().tolist()
    subs_unmatched_vpus = df_cwt_unmatched["vpuid"].unique().tolist()
    df_cwt_matched = df_cwt[df_cwt["nearest_dist_m"].isna()]
    counts = df_cwt_matched["divide_id"].value_counts()
    cats_sub1 = counts[counts == 1].index.unique()
    cats_sub2 = counts[counts == 2].index.unique()
    cats_sub3 = counts[counts == 3].index.unique()
    cats_other = counts[counts > 3].index.unique()
    print(f"Total number of catchments in {domain.upper()}: {len(cats_total)}")
    print(
        f"Number of catchments matched with nearest neighor (i.e., no overlapping): {len(cats_unmatched)}, {round(len(cats_unmatched) / len(cats_total) * 100, 2)}%"
    )
    print(
        f"Number of catchments mapped to 1 {attr.upper()} subbain: {len(cats_sub1)}, {round(len(cats_sub1) / len(cats_total) * 100, 2)}%"
    )
    print(
        f"Number of catchments mapped to 2 {attr.upper()} subbains: {len(cats_sub2)}, {round(len(cats_sub2) / len(cats_total) * 100, 2)}%"
    )
    print(
        f"Number of catchments mapped to 3 {attr.upper()} subbains: {len(cats_sub3)}, {round(len(cats_sub3) / len(cats_total) * 100, 2)}%"
    )
    print(
        f"Number of catchments mapped to more than 3 {attr.upper()} subbains: {len(cats_other)}, {round(len(cats_other) / len(cats_total) * 100, 2)}%"
    )

    print("\nSummary statistics of overlap coverage for matched catchments:")
    # Compute accumulated overlap for each divide_id
    accum_overlap = df_cwt_matched.groupby("divide_id")["overlap_percentage"].sum()

    # Compute summary statistics on the accumulated values
    summary = accum_overlap.describe()
    print(summary)

    # save nearest neighbor unmatched catchments and subbasins to file
    if len(cats_unmatched) > 0:
        unmatched_file = Path(
            output_dir,
            f"ngen_catchments_nearest_{attr}_subbasins_{domain.lower()}.parquet",
        ).expanduser()
        print(
            f"\nSaving nearest neighbor unmatched catchments and subbasins to {unmatched_file}..."
        )
        df_cwt_unmatched.to_parquet(
            unmatched_file,
            engine="pyarrow",
            index=False,
        )

    # plot nearest neighbor matches
    if len(cats_unmatched) > 0:
        print("\nPlotting nearest neighbor matched catchments and subbasins...")
        plot_cats_subs(
            attr=attr,
            cats=cats_unmatched,
            subs=subs_unmatched,
            output_dir=output_dir,
            vpu_shp_dir=vpu_shp_dir,
            vpus=subs_unmatched_vpus,
            domain=domain,
        )


def plot_cats_subs(
    attr: str,
    cats: list,
    subs: list,
    output_dir: Path = None,
    vpu_shp_dir: str | Path = None,
    vpus: list = None,
    domain: str = "conus",
):
    """Plot NextGen catchments and corresponding attribute dataset subbasins."""
    if not cats:
        print("No catchments to plot.")
        return

    print(
        "Get corresponding NextGen catchments and attribute dataset sub-basins for plotting..."
    )
    gdf_cats = gpd.GeoDataFrame()
    gdf_subs = gpd.GeoDataFrame()
    for vpu in vpus:
        print(f"Processing {vpu}...")

        # get NextGen catchments for the vpu
        shp_vpu = get_vpu_gpd(vpu, vpu_shp_dir)
        shp_vpu = shp_vpu[shp_vpu["divide_id"].isin(set(cats))].copy()
        gdf_cats = pd.concat([gdf_cats, shp_vpu], ignore_index=True)
        del shp_vpu
        gc.collect()

        # get attribute dataset sub-basins for the vpu
        attr_dict = get_attr_info(attr, vpu=vpu)
        gdf_all = gpd.read_file(
            Path(attr_dict["file"]),
            layer=attr_dict["layer"],
            columns=[attr_dict["id_column"], "geometry"],
        )
        gdf_subs_vpu = gdf_all[gdf_all[attr_dict["id_column"]].isin(set(subs))].copy()
        gdf_subs = pd.concat([gdf_subs, gdf_subs_vpu], ignore_index=True)
        del gdf_all, gdf_subs_vpu
        gc.collect()

    if gdf_cats.empty:
        print("No NextGen catchments found for plotting.")
        return
    if gdf_subs.empty:
        print("No attribute dataset subbasins found for plotting.")
        return

    # plot
    fig, ax = plt.subplots(figsize=(12, 6))
    gdf_subs = gdf_subs.to_crs(epsg=4326)
    gdf_cats = gdf_cats.to_crs(epsg=4326)

    gdf_subs.plot(ax=ax, color="lightgray", edgecolor="black", linewidth=5.0)
    gdf_cats.plot(ax=ax, color="none", edgecolor="red", linewidth=1.0)
    plt.title(
        f"{domain.upper()} NextGen catchments (red) and corresponding nearest-neighbour {attr_dict['name'].upper()} subbasins (black)"
    )

    # save plot
    plot_file = Path(
        output_dir,
        f"ngen_catchments_nearest_{attr_dict['name']}_subbasins_{domain}.png",
    ).expanduser()
    plot_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"Plot saved to {plot_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create NextGen catchment to attribute dataset subbasin crosswalk."
    )
    parser.add_argument(
        "--attr",
        type=str,
        required=True,
        help="Attribute dataset to create crosswalk for (options: hydroatlas, hlr, streamcat/nhdplus).",
    )

    parser.add_argument(
        "--domain",
        type=str,
        default="conus",
        help="Domain to process (default: conus).",
    )
    parser.add_argument(
        "--cwt_dir",
        type=str,
        default="~/repos/nwm-region-mgr/data/inputs",
        help="Output directory to save crosswalk tables.",
    )
    parser.add_argument(
        "--vpu_gpkg_dir",
        type=str,
        default="~/repos/nwm-region-mgr/data/inputs/hydrofabric/vpu_divides",
        help="Directory containing hydrofabric data for vpu divides.",
    )

    parser.add_argument(
        "--threshold_max",
        type=float,
        default=99.0,
        help="Maximum overlap percentage threshold to consider a subbasin matched (default: 99.0).",
    )
    parser.add_argument(
        "--threshold_min",
        type=float,
        default=20.0,
        help="Minimum overlap percentage threshold to consider a subbasin matched (default: 20.0).",
    )

    args = parser.parse_args()
    cwt_dir = Path(args.cwt_dir).expanduser()
    cwt_dir.mkdir(parents=True, exist_ok=True)

    df_cwt = process_cwt_domain(
        attr=args.attr,
        domain=args.domain,
        cwt_dir=cwt_dir,
        shp_dir=args.vpu_gpkg_dir,
        threshold_max=args.threshold_max,
        threshold_min=args.threshold_min,
    )

    if not df_cwt.empty:
        print(
            f"Crosswalk creation completed for {args.attr} for the {args.domain} domain."
        )
