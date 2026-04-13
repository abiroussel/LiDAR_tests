# LiDAR_tests

## Repo tree

```bash
├── data
│   ├── images
│   ├── lidar
│   ├── s1
│   │   ├── data.npy
│   │   ├── ids.npy
│   │   └── labels.npy
│   ├── s2
│   │   ├── data.npy
│   │   ├── ids.npy
│   │   └── labels.npy
│   └── vector_data
│       ├── polygons.gpkg
│       └── roi.gpkg
├── download_lidar.ipynb
├── download_utils.py
├── environment.yml
├── main.ipynb
├── MLP.py
├── README.MD
├── texture_encoders.py
└── utils.py
```

## Downloading LiDAR data
**<u>Context and goal:</u>** This code was implemented to get LiDAR images for vineayard parcels. The objective is to get one image for eache parcel. 

**<u>Data source:</u>** The LiDAR data is produced by the French IGN. LiDAR point clouds are supposed to be acquired for all of the French territory over the period from 2021 to 2026. At the moment, each region has only been flown over once. Therefore, there may be some wide differences between acquisition dates from one place to another.

[Information about the IGN LiDAR program](https://www.ign.fr/institut/programme-lidar-hd-vers-une-nouvelle-cartographie-3d-du-territoire)

[Downloading LiDAR data online](https://cartes.gouv.fr/telechargement/IGNF_MNT-LIDAR-HD)


### Visualize acquisition metadata
Since the span of acquisition dates is large, the user may want to select the data based on the available dates of acquisition. 

### Download images

One of the three following data types can be selected to be downloaded. All these products are derived from the LiDAR points and are 1-band .tif images with a resolution of 0.5m. Each tile has a size of 1km x 1km.

| Data type (FR) | Data type (EN) | Definition|
|-|-|-|
MNS (Modèle Numérique de Surface) | DSM | Measures the elevation of the soil AND any other natural or artificial object (trees, buildings...)|
MNT (Modèle Numérique de Terrain) | DEM | Measures the elevation of the terrain (only bare soil) |
MNH  (Modèle Numérique de Hauteur) | Above ground heights | Measures the difference between MNS and MNT (to get the height of objects)|

### Crop polygon images

Given a database of polygons, the corresponding LiDAR product images are outlined. They are saved as 1-band .jpg images for easier loading with standard convolution networks. 
