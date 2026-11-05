"""Extract the small arrays the figure's map panels need from the course data.

Run once in the geostats2026 conda env (needs xarray + netCDF4):
    python prep_map_data.py <path to GEO/data>            # Rubble Glacier
    python prep_map_data.py <path to GEO/data> ase        # Amundsen Sea Embayment
Writes data/antarctica_mask_5km.npz and data/study_area.npz (Rubble) or
data/study_area_<region>.npz next to this script.

The study area is cropped exactly as in Lab 1 (Rubble Glacier box by default),
straight from BedMachine and the Bedmap3 thickness compilation.
"""
import sys
from pathlib import Path

import numpy as np
import xarray as xr

# Boxes in polar stereographic (EPSG:3031) metres, on Bedmap3's 500 m cell centres (...250/...750)
REGIONS = {
    # Lab 1 box
    "rubble": (-2_000_250, -1_800_250, 650_250, 850_250),
    # Amundsen Sea Embayment: Thwaites + Pine Island glaciers, 650 x 700 km
    "ase": (-1_900_250, -1_250_250, -750_250, -50_250),
}
RES = 500

src = Path(sys.argv[1])
region = sys.argv[2] if len(sys.argv) > 2 else "rubble"
XMIN, XMAX, YMIN, YMAX = REGIONS[region]
out = Path(__file__).with_name("data")
out.mkdir(exist_ok=True)

bm = xr.open_dataset(src / "BedMachineAntarctica-v3.nc")

# Continent-wide mask for the locator map (every 10th 500 m cell -> 5 km)
m = bm["mask"].isel(x=slice(None, None, 10), y=slice(None, None, 10)).load()
np.savez_compressed(out / "antarctica_mask_5km.npz",
                    x=m.x.values, y=m.y.values, mask=m.values.astype("int8"))

# Study area, same steps as Lab 1
x_slice = slice(XMIN, XMAX) if bm.x[0] < bm.x[-1] else slice(XMAX, XMIN)
y_slice = slice(YMIN, YMAX) if bm.y[0] < bm.y[-1] else slice(YMAX, YMIN)
glacier = bm.sel(x=x_slice, y=y_slice)[["mask", "surface", "bed"]].load()

ant = np.arange(-3_333_250, 3_333_250 + RES, RES)
xi = [int(np.where(ant == v)[0][0]) for v in (XMIN, XMAX)]
yi = [int(np.where(ant == v)[0][0]) for v in (YMIN, YMAX)]
thick = np.load(src / "bedmap3_raw_thick_compilation.npy", mmap_mode="r")
thick = np.flip(thick, axis=0)[yi[0]:yi[1], xi[0]:xi[1]]
assert thick.shape == glacier["surface"].shape, (thick.shape, glacier["surface"].shape)

mask = glacier["mask"].values.astype("int8")
radar_bed = glacier["surface"].values - np.flipud(thick)
radar = np.isfinite(radar_bed) & (mask == 2)
# On ice-free rock (mask 1) the bed is the surface: exact conditioning data.
rock = mask == 1
xx, yy = np.meshgrid(glacier.x.values, glacier.y.values)
np.savez_compressed(out / ("study_area.npz" if region == "rubble" else f"study_area_{region}.npz"),
                    x=glacier.x.values, y=glacier.y.values, mask=mask,
                    radar_x=xx[radar], radar_y=yy[radar],
                    radar_bed=radar_bed[radar].astype("float32"),
                    rock_x=xx[rock], rock_y=yy[rock],
                    rock_bed=glacier["surface"].values[rock].astype("float32"),
                    # For comparison only, never used as conditioning data.
                    bedmachine_bed=glacier["bed"].values.astype("float32"))
print("grid", mask.shape, "mask classes", np.unique(mask),
      "| radar cells on grounded ice:", int(radar.sum()), "| rock cells:", int(rock.sum()),
      "| bed range", float(np.nanmin(radar_bed[radar])), float(np.nanmax(radar_bed[radar])))
