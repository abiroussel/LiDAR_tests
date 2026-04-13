import json
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import requests
from pathlib import Path
import rasterio as rio
from rasterio.mask import mask
from PIL import Image
from typing import Dict


def get_years(row: pd.Series) -> list[int]:
    """
    Extract all years covered by a time interval.

    Parameters
    ----------
    row : pd.Series
        A row containing at least:
        - 'start': pd.Timestamp
        - 'end': pd.Timestamp

    Returns
    -------
    list[int]
        List of years between start and end (inclusive).
    """
    return list(range(row["start"].year, row["end"].year + 1))


def unroll_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand a column containing JSON-like metadata strings into separate columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a 'metadata' column (stringified JSON).

    Returns
    -------
    pd.DataFrame
        DataFrame with metadata fields expanded into new columns.
    """
    # Convert JSON string → dictionary
    df["metadata_dict"] = df["metadata"].apply(json.loads)

    # Expand dictionary into columns
    metadata_expanded = df["metadata_dict"].apply(pd.Series)

    # Merge back and drop original columns
    df = pd.concat(
        [df.drop(columns=["metadata", "metadata_dict"]), metadata_expanded],
        axis=1
    )

    return df


def plot_acquisition_dates(df: pd.DataFrame) -> None:
    """
    Plot acquisition timelines per year as Gantt-like charts.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain:
        - 'datedebut': start date (string or datetime)
        - 'datefin': end date (string or datetime)

    Returns
    -------
    None
        Saves figures to disk and displays them.
    """

    # Combine start/end into a single string column
    df['aquisition_dates'] = df['datedebut'].astype(str) + ' / ' + df['datefin'].astype(str)

    # Extract unique intervals
    dates = pd.DataFrame()
    dates[['start', 'end']] = pd.Series(df['aquisition_dates'].unique()).str.split(' / ', expand=True)

    # Convert to datetime
    dates['start'] = pd.to_datetime(dates['start'])
    dates['end'] = pd.to_datetime(dates['end'])

    # Check if intervals span multiple years
    dates["same_year"] = dates["start"].dt.year == dates["end"].dt.year

    if not dates["same_year"].all():
        raise ValueError("Some intervals span multiple years!")
    else:
        dates["year"] = dates["start"].dt.year

    # Plot per year
    for year, group in dates.groupby("year"):

        start_year = pd.Timestamp(f"{year}-01-01")
        end_year = pd.Timestamp(f"{year}-12-31")

        fig, ax = plt.subplots(figsize=(12, 7))

        # Plot horizontal bars (Gantt style)
        for i, (idx, row) in enumerate(group.iterrows()):
            ax.barh(
                y=i,
                width=(row["end"] - row["start"]).days,
                left=mdates.date2num(row["start"]),
                height=0.6,
                color="#378ADD",
                edgecolor="#185FA5",
                linewidth=0.5,
            )

        # Force full-year display
        ax.set_xlim(start_year, end_year)

        # Y-axis labels
        ax.set_yticks(range(len(group)))
        ax.set_yticklabels(group.index, fontsize=10)

        # X-axis formatting
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))

        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Tile number", fontsize=11)
        ax.set_title(f"Acquisition timeline — {year}", fontsize=13)

        ax.grid(axis="x", color="grey", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"figures/acquisition_timeline_{year}.png", dpi=150)
        plt.show()


def get_year_data(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    """
    Split a DataFrame into subsets by year extracted from 'datedebut'.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'datedebut' column formatted as 'YYYY-MM-DD'.

    Returns
    -------
    dict[int, pd.DataFrame]
        Dictionary mapping year → subset DataFrame.
    """
    df['year'] = df['datedebut'].str.split('-').str[0].astype(int)

    year_tiles = {}
    for year in df['year'].unique():
        year_tiles[year] = df[df['year'] == year]

    return year_tiles


def get_product_urls(
    tiles_df: gpd.GeoDataFrame,
    product: str,
    roi: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Retrieve product URLs from IGN WFS and join them to tile metadata.

    Parameters
    ----------
    tiles_df : gpd.GeoDataFrame
        Tile metadata with 'name' column.
    product : str
        Product name (e.g., 'MNH', 'MNT', etc.).
    roi : gpd.GeoDataFrame
        Region of interest.

    Returns
    -------
    gpd.GeoDataFrame
        Updated tiles_df with URL column for the requested product.
    """

    if product != 'LIDAR':

        # Extract bounding box of ROI
        minx, miny, maxx, maxy = roi.total_bounds

        # Build WFS request
        product_url = (
            "https://data.geopf.fr/wfs/ows?"
            "service=WFS&version=2.0.0&request=GetFeature"
            f"&typeName=IGNF_{product}-LIDAR-HD:dalle"
            "&outputFormat=application/json"
            f"&bbox={minx},{miny},{maxx},{maxy},{roi.crs.to_string()}"
        )

        # Load WFS data
        wfs_gdf = gpd.read_file(product_url)
        wfs_gdf = wfs_gdf.to_crs(roi.crs)

        # Spatial join with ROI
        product_df = gpd.sjoin(wfs_gdf, roi, predicate="intersects")

        # Create ID key
        product_df['id_name'] = product_df.name.str[:17]
        product_df.set_index('id_name', inplace=True)

        tiles_df['id_name'] = tiles_df.name.str[:17]
        tiles_df.set_index('id_name', inplace=True)

        # Join URLs
        tiles_df = tiles_df.join(product_df['url'], rsuffix=f'_{product}', how='inner')

    return tiles_df


def download_tiles(
    tiles_df: pd.DataFrame,
    product: str,
    lidar_dir: str
) -> pd.DataFrame:
    """
    Download raster tiles from URLs.

    Parameters
    ----------
    tiles_df : pd.DataFrame
        Must contain column 'url_<product>'.
    product : str
        Product name.
    lidar_dir : str
        Output directory.

    Returns
    -------
    pd.DataFrame
        Same DataFrame after download.
    """

    for id_ in tiles_df.index:

        response = requests.get(tiles_df.loc[id_, f'url_{product}'], stream=True)

        with open(f"{lidar_dir}/{id_}.tif", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    return tiles_df


def get_polygon_images(
    polygons_path: str,
    lidar_dir: str,
    image_dir: str,
    tiles_df: gpd.GeoDataFrame,
    index_col: str
) -> None:
    """
    Extract raster images corresponding to polygons and save as JPG.

    Parameters
    ----------
    polygons_path : str
        Path to polygon GeoPackage.
    lidar_dir : str
        Directory containing raster tiles.
    image_dir : str
        Output directory for JPG images.
    tiles_df : gpd.GeoDataFrame
        Tile metadata (index = image names).
    index_col : str
        Column to use as polygon identifier.

    Returns
    -------
    None
    """

    lidar_dir = Path(lidar_dir)

    # Prepare image footprints
    images_df = tiles_df.copy()
    if 'index_right' in images_df.columns:
        images_df.drop(columns=['index_right'], inplace=True)
    images_df['image_name'] = images_df['name'].str[:17]

    # Load polygons
    polygons = gpd.read_file(polygons_path).set_index(index_col)

    # Spatial join
    matches = gpd.sjoin(polygons, images_df, predicate="intersects")

    # Build file paths
    matches["filepath"] = matches["image_name"].apply(lambda x: lidar_dir / f"{x}.tif")

    output_dir = Path(image_dir)
    output_dir.mkdir(exist_ok=True)

    # Process each raster only once
    for filepath, group in matches.groupby("filepath"):
        try: 
            with rio.open(filepath) as src:

                for idx, row in group.iterrows():

                    geom = [row["geometry"]]

                    # Clip raster
                    out_image, _ = mask(src, geom, crop=True)

                    data = out_image[0]

                    # Handle nodata
                    if src.nodata is not None:
                        data = np.where(data == src.nodata, np.nan, data)

                    # Normalize to 0–255
                    data_min = np.nanmin(data)
                    data_max = np.nanmax(data)

                    if data_max > data_min:
                        data_norm = (data - data_min) / (data_max - data_min)
                    else:
                        data_norm = np.zeros_like(data)

                    data_uint8 = (data_norm * 255).astype(np.uint8)

                    # Convert to image
                    img = Image.fromarray(data_uint8)

                    # Save JPG
                    out_path = output_dir / f"{idx}.jpg"
                    img.save(out_path, "JPEG")

        except Exception:
            print(f'Could not extract image for polygon {idx}')
            pass