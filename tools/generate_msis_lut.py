# tools/generate_msis_lut.py
"""
Precompute an MSIS density LUT for (alt_km, f107, ap).
- Uses PyMSIS (MSIS2) with f107 (daily) and f107a (81-day avg ~= f107 if none handy).
- Produces docs/msis_lut.json with shape [len(alt)][len(f107)][len(ap)].
- Keep grids modest so the JSON stays small enough for GitHub Pages.

pip install pymsis numpy tqdm

Run from repo root:
  python tools/generate_msis_lut.py
"""

import json
import datetime as dt
from pathlib import Path

import numpy as np
from tqdm import tqdm
import pymsis

# --- pymsis version compatibility ---
# Supports both newer (pymsis.calculate) and older (pymsis.msis.run) APIs.
def msis_eval(dates, lons, lats, alts, f107s, f107as, aps):
    """Call PyMSIS in a version-agnostic way.
    Arguments follow the documented order: dates, lons, lats, alts, f107s, f107as, aps.
    Returns an ndarray whose last dimension contains variables; total mass density is [..., 0].
    """
    if hasattr(pymsis, "calculate"):
        # pymsis >= 0.11
        return pymsis.calculate(dates, lons, lats, alts, f107s, f107as, aps)
    else:
        # pymsis <= 0.10
        from pymsis import msis as _msis
        return _msis.run(dates, lons, lats, alts, f107s, f107as, aps)

# ----------------------------
# Grids — densified for better interpolation (still lightweight JSON)
# ----------------------------
ALT_KM = np.arange(300.0, 701.0, 10.0, dtype=float)            # 300..700 km @ 10 km
F107   = np.array([60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 200, 220, 240, 260, 280, 300], dtype=float)
AP     = np.array([0, 4, 7, 10, 15, 20, 30, 40, 60, 80, 100, 120, 160, 200, 260, 320, 400], dtype=float)

# Reference timestamp (consistent with your notebook)
REF_TIME = dt.datetime(2023, 4, 24, 12, 0, 0, tzinfo=dt.timezone.utc)

# ----------------------------
# MSIS orbit-average
# ----------------------------

def msis_density_orbit_avg(alt_km, f107, ap, inc_deg=90.0, samples=240, t0=REF_TIME):
    """
    Orbit-averaged neutral mass density [kg/m^3] over one circular orbit at altitude alt_km.
    - Uses a near-polar inclination by default (inc_deg=90°).
    - Samples `samples` points uniformly around the orbit and averages MSIS density.
    - Each sample updates time (to capture local-time dependence) and sweeps longitude.
    """
    # Orbital period for circular orbit at altitude alt_km
    MU = 3.986004418e14  # m^3/s^2
    RE = 6371e3          # m
    r  = RE + alt_km*1e3
    T  = 2.0*np.pi*np.sqrt(r**3/MU)  # seconds

    # Build sample arrays
    k = np.arange(samples, dtype=float)
    frac = k / samples
    # Argument of latitude u in [0, 2π)
    u = 2.0*np.pi*frac

    # Latitude for circular orbit: lat = asin( sin(i) * sin(u) )
    inc = np.deg2rad(inc_deg)
    lat_rad = np.arcsin(np.sin(inc) * np.sin(u))
    lat_deg = np.rad2deg(lat_rad)

    # Sweep longitudes uniformly; advance UTC across the orbit
    lon_deg = (frac * 360.0) % 360.0
    times   = np.array([t0 + dt.timedelta(seconds=float(T*fi)) for fi in frac])

    # Vectorized MSIS call (documented order): dates, lons, lats, alts, f107s, f107as, aps
    nd = samples
    dates  = times
    lons   = lon_deg.astype(float)
    lats   = lat_deg.astype(float)
    alts   = np.full(nd, alt_km, dtype=float)
    f107s  = np.full(nd, f107, dtype=float)
    f107as = np.full(nd, f107, dtype=float)  # proxy for 81-day average unless you supply it
    # aps must be 7 values per date; replicate the same Ap across all 7 indices
    aps    = np.tile(np.array([ap]*7, dtype=float), (nd, 1))

    out = msis_eval(dates, lons, lats, alts, f107s, f107as, aps)
    out = np.squeeze(out)
    # Total mass density is in the first variable slot
    rho_samples = out[..., 0]
    return float(np.mean(rho_samples))


def main():
    out_path = Path("docs/msis_lut.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rho = np.zeros((len(ALT_KM), len(F107), len(AP)), dtype=float)

    # Loop
    for i, akm in enumerate(tqdm(ALT_KM, desc="alt_km")):
        for j, f in enumerate(F107):
            for k, a in enumerate(AP):
                rho[i, j, k] = msis_density_orbit_avg(akm, f, a)

    payload = {
        "meta": {
            "model": "MSIS2 via pymsis",
            "generated_utc": dt.datetime.utcnow().isoformat() + "Z",
            "ref_time_utc": REF_TIME.isoformat().replace("+00:00", "Z"),
            "lat_deg": 0.0,
            "lon_deg": 0.0,
            "cd_assumed": 2.2,
            "orbit": "circular",
            "inclination_deg": 90.0,
            "note": "Orbit-averaged density over one circular polar orbit (inc≈90°) using samples=240; f107a≈f107; version-agnostic PyMSIS call."
        },
        "grid": {
            "alt_km": ALT_KM.tolist(),
            "f107": F107.tolist(),
            "ap": AP.tolist()
        },
        "rho": rho.tolist()  # [alt][f107][ap]
    }

    with out_path.open("w") as f:
        json.dump(payload, f, separators=(",", ":"), indent=None)  # compact JSON

    size_mb = out_path.stat().st_size / (1024*1024)
    print(f"Wrote {out_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()