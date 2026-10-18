from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

try:
    import h5py
except ImportError:
    h5py = None


# Physical constants, empirical limits, and fixed analysis settings.
PHYSICS = {
    "g": 9.81,
    "kappa": 0.41,
    "ref_a_ratio": 0.001,
    "z0_ratio": 0.001,
    "sample_height_ratio_baseline": 0.90,
    "sample_height_ratio_range": (0.60, 0.95),
    "n_channel_baseline": 0.035,
    "n_channel_range": (0.024, 0.075),
    "slope_min_valid_m_per_km": 1.0e-6,
    "slope_max_valid_m_per_km": 100.0,
    "profile_min_points": 3,
    "profile_min_r2": 0.50,
    "profile_min_accepted_final": 5,
    "P_min": 1.0e-4,
    "P_support_max": 0.50,
    "P_grid_size": 500,
    "eta_grid_size": 101,
    "vertical_grid_size": 2500,
    "minimum_supported_ensemble_fraction": 0.80,
    "f_pp_baseline": 0.60,
    "f_pp_scenario_range": (0.30, 0.90),
    "ensemble_size": 300,
    "random_seed": 42,
    "chunk_size": 5000,
    "minimum_discharge_match_rate_final": 0.90,
    "maximum_manning_relative_residual": 1.0e-6,
}

ACCEPTED_Q_COLUMNS = ("mean_q", "meanDRS_Q", "Q_m3s", "discharge_m3s")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute river-reach K_PP and K_TP with empirical Rouse applicability screening."
    )
    parser.add_argument("--sword", required=True, help="SWORD reach table (CSV, XLSX, Parquet)")
    discharge = parser.add_mutually_exclusive_group(required=True)
    discharge.add_argument("--discharge", help="Reach-scale discharge table")
    discharge.add_argument("--sos-prior-dir", help="Directory with continent SoS prior NetCDF files")
    parser.add_argument("--profiles", required=True, help="Depth-resolved calibration/validation table")
    parser.add_argument("--fpp", help="Optional paired phosphorus data for empirical f_PP")
    parser.add_argument("--hydraulic-benchmark", help="Optional depth benchmark table")
    parser.add_argument("--tp-benchmark", help="Optional paired single/integrated TP table")
    parser.add_argument("--outdir", default="results", help="Output directory (default: results)")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional short-run check")
    return parser.parse_args()


def load_table(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    ext = p.suffix.lower()
    if ext in (".xlsx", ".xls"):
        return pd.read_excel(p, nrows=nrows)
    if ext in (".csv", ".txt"):
        return pd.read_csv(p, nrows=nrows)
    if ext == ".parquet":
        d = pd.read_parquet(p)
        return d.head(nrows) if nrows is not None else d
    raise ValueError(f"Unsupported table format: {ext}")

def normalize_reach_id(s: pd.Series) -> pd.Series:
    """Keep long SWORD identifiers exact and avoid scientific-notation / trailing-.0 errors."""
    def one(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)) and np.isfinite(v):
            return str(int(v)) if float(v).is_integer() else format(v, ".0f")
        t = str(v).strip()
        if t.endswith(".0") and t[:-2].isdigit():
            t = t[:-2]
        return t
    return s.map(one)

def validate_sword_source(df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    required = {"reach_id", "width", "slope"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"SWORD source is missing required fields: {sorted(missing)}")

    out = df.copy()
    out["reach_id"] = normalize_reach_id(out["reach_id"])
    out["width"] = pd.to_numeric(out["width"], errors="coerce")
    out["slope"] = pd.to_numeric(out["slope"], errors="coerce")

    duplicate_ids = int(out["reach_id"].duplicated(keep=False).sum())
    missing_ids = int(out["reach_id"].isna().sum())
    nonpositive_width = int((out["width"] <= 0).sum())

    smin = cfg["slope_min_valid_m_per_km"]
    smax = cfg["slope_max_valid_m_per_km"]
    out["slope_valid"] = np.isfinite(out["slope"]) & (out["slope"] > smin) & (out["slope"] <= smax)
    out["width_valid"] = np.isfinite(out["width"]) & (out["width"] > 0)

    if "type" in out.columns:
        type_values = ", ".join(map(str, sorted(pd.unique(out["type"].dropna()))[:20]))
        type1_n = int((pd.to_numeric(out["type"], errors="coerce") == 1).sum())
    else:
        type_values = "not supplied"
        type1_n = np.nan

    qa = pd.DataFrame([
        ["source_rows", len(out)],
        ["unique_reach_id", out["reach_id"].nunique(dropna=True)],
        ["duplicate_id_rows", duplicate_ids],
        ["missing_reach_id", missing_ids],
        ["nonpositive_width", nonpositive_width],
        [f"slope_le_{smin:g}_m_per_km", int((out["slope"] <= smin).sum())],
        [f"slope_gt_{smax:g}_m_per_km", int((out["slope"] > smax).sum())],
        ["primary_slope_valid_rows", int(out["slope_valid"].sum())],
        ["type_values", type_values],
        ["type_1_rows", type1_n],
    ], columns=["metric", "value"])

    if duplicate_ids > 0:
        raise ValueError("Duplicate SWORD reach_id values found in source table.")
    return out, qa


def load_sos_prior_folder(folder: str):
    """Read all continent SoS-priors NetCDF files and return reach_id + mean_q."""
    if not folder or not Path(folder).exists():
        raise FileNotFoundError(f"SoS prior folder not found: {folder}")
    if h5py is None:
        raise ImportError("Reading SoS NetCDF priors requires h5py: pip install h5py")

    files = sorted(Path(folder).glob("*_sword_v17*_SOS_priors.nc"))
    if not files:
        raise FileNotFoundError(f"No *_sword_v17*_SOS_priors.nc files in {folder}")

    parts = []
    qa_rows = []
    for p in files:
        continent = p.name.split("_")[0].lower()
        with h5py.File(p, "r") as f:
            if "reaches" not in f or "model" not in f:
                raise ValueError(f"{p.name}: missing /reaches or /model")
            rid = np.asarray(f["reaches"]["reach_id"][()]).astype(np.int64)
            qd = f["model"]["mean_q"]
            q = np.asarray(qd[()]).astype(float)
            fill = qd.attrs.get("_FillValue", -999999999999.0)
            fill = float(np.asarray(fill).reshape(-1)[0])
            valid = np.isfinite(q) & (q != fill) & (q > 0)

            d = pd.DataFrame({
                "reach_id": rid.astype(str),
                "mean_q": np.where(valid, q, np.nan),
                "prior_continent": continent,
                "prior_file": p.name,
            })
            parts.append(d)

            source_attr = f.attrs.get("source", "")
            if isinstance(source_attr, bytes):
                source_attr = source_attr.decode("utf-8", "ignore")
            qa_rows.append({
                "continent": continent,
                "file": p.name,
                "n_prior_reaches": len(d),
                "n_valid_mean_q": int(valid.sum()),
                "valid_mean_q_fraction": float(valid.mean()),
                "source_attribute": str(source_attr),
            })

    prior = pd.concat(parts, ignore_index=True)
    if prior["reach_id"].duplicated().any():
        raise ValueError("Duplicate reach_id encountered across SoS-prior files.")
    return prior, pd.DataFrame(qa_rows)


def join_discharge_prior(source: pd.DataFrame, table_path: Optional[str],
                         sos_dir: Optional[str], cfg: dict):
    """Match independent discharge estimates to SWORD reaches by reach_id."""
    if sos_dir:
        prior, qa_files = load_sos_prior_folder(sos_dir)
        qcol = "mean_q"
        columns = ["reach_id", "mean_q", "prior_continent", "prior_file"]
        min_rate = 0.80  # Original NetCDF-branch minimum.
        source_label = "SoS NetCDF"
    else:
        prior = load_table(table_path)
        qa_files = pd.DataFrame()
        qcol = next((c for c in ACCEPTED_Q_COLUMNS if c in prior.columns), None)
        if qcol is None:
            raise ValueError(f"Discharge table needs one of {ACCEPTED_Q_COLUMNS}.")
        columns = ["reach_id", qcol]
        min_rate = cfg["minimum_discharge_match_rate_final"]
        source_label = "discharge table"
    if "reach_id" not in prior.columns:
        raise ValueError("Discharge data must contain reach_id.")
    prior["reach_id"] = normalize_reach_id(prior["reach_id"])
    prior[qcol] = pd.to_numeric(prior[qcol], errors="coerce")
    if prior["reach_id"].duplicated().any():
        if sos_dir:
            raise ValueError("Duplicate reach_id values in SoS priors.")
        prior = prior.drop_duplicates("reach_id", keep="first")

    out = source.merge(prior[columns], on="reach_id", how="left", validate="one_to_one")
    out["Q_m3s"] = pd.to_numeric(out[qcol], errors="coerce")
    out["Q_valid"] = np.isfinite(out["Q_m3s"]) & (out["Q_m3s"] > 0)
    matched, total = int(out["Q_valid"].sum()), len(out)
    rate = matched / total if total else np.nan
    if rate < min_rate:
        raise ValueError(f"Only {rate:.2%} of reaches have valid discharge; required {min_rate:.0%}.")
    good_q = out.loc[out["Q_valid"], "Q_m3s"]
    qa = pd.DataFrame([
        ["discharge_source", source_label],
        ["source_reaches", total],
        ["matched_positive_Q", matched],
        ["unmatched_or_invalid_Q", total - matched],
        ["match_rate", rate],
        ["Q_median_m3s", float(good_q.median()) if matched else np.nan],
        ["Q_p05_m3s", float(good_q.quantile(0.05)) if matched else np.nan],
        ["Q_p95_m3s", float(good_q.quantile(0.95)) if matched else np.nan],
    ], columns=["metric", "value"])
    return out, qa, qa_files


def manning_rectangular_depth(Q, W, S, n_channel, max_iter=20):
    """Solve Q=(1/n) A Rh^(2/3) S^(1/2), A=WH, Rh=WH/(W+2H)."""
    Q = np.asarray(Q, dtype=float)
    W = np.asarray(W, dtype=float)
    S = np.asarray(S, dtype=float)
    H = ((np.maximum(Q, 1e-12) * n_channel) /
         (np.maximum(W, 1e-9) * np.sqrt(np.maximum(S, 1e-15)))) ** 0.6
    H = np.clip(H, 1e-3, 1e4)

    for _ in range(max_iter):
        A = W * H
        Rh = A / (W + 2.0 * H)
        qhat = A * Rh ** (2/3) * np.sqrt(S) / n_channel
        dlnq_dH = 1/H + (2/3) * (1/H - 2/(W + 2*H))
        dq_dH = np.maximum(qhat * dlnq_dH, 1e-12)
        Hnew = np.clip(H - (qhat - Q) / dq_dH, 1e-3, 1e4)
        if np.nanmax(np.abs(Hnew-H)/np.maximum(H, 1e-12)) < 1e-10:
            H = Hnew
            break
        H = Hnew

    A = W * H
    Rh = A / (W + 2.0 * H)
    qhat = A * Rh ** (2/3) * np.sqrt(S) / n_channel
    resid = np.abs(qhat-Q)/np.maximum(Q, 1e-12)
    return H, Rh, resid

def reconstruct_hydraulics(df: pd.DataFrame, cfg: dict):
    out = df.copy()
    out["S_m_per_m"] = out["slope"] / 1000.0
    valid = out["slope_valid"] & out["width_valid"] & out["Q_valid"]
    out["hydraulic_input_valid"] = valid

    n0 = cfg["n_channel_baseline"]
    nlo, nhi = cfg["n_channel_range"]

    for prefix, nval in (("base", n0), ("nlo", nlo), ("nhi", nhi)):
        H = np.full(len(out), np.nan)
        Rh = np.full(len(out), np.nan)
        res = np.full(len(out), np.nan)
        if valid.any():
            idx = np.where(valid.to_numpy())[0]
            H[idx], Rh[idx], res[idx] = manning_rectangular_depth(
                out.loc[valid, "Q_m3s"].to_numpy(),
                out.loc[valid, "width"].to_numpy(),
                out.loc[valid, "S_m_per_m"].to_numpy(),
                nval,
            )
        out[f"H_{prefix}_m"] = H
        out[f"Rh_{prefix}_m"] = Rh
        out[f"manning_resid_{prefix}"] = res
        out[f"u_star_{prefix}_m_s"] = np.sqrt(cfg["g"] * Rh * out["S_m_per_m"].to_numpy())

    out["hydraulic_solution_valid"] = (
        out["hydraulic_input_valid"] &
        np.isfinite(out["H_base_m"]) & (out["H_base_m"] > 0) &
        np.isfinite(out["u_star_base_m_s"]) & (out["u_star_base_m_s"] > 0) &
        (out["manning_resid_base"] <= cfg["maximum_manning_relative_residual"])
    )

    q = pd.DataFrame([
        ["hydraulic_input_valid", int(out["hydraulic_input_valid"].sum())],
        ["hydraulic_solution_valid", int(out["hydraulic_solution_valid"].sum())],
        ["H_median_m", float(np.nanmedian(out.loc[out["hydraulic_solution_valid"], "H_base_m"]))],
        ["H_p05_m", float(np.nanpercentile(out.loc[out["hydraulic_solution_valid"], "H_base_m"], 5))],
        ["H_p95_m", float(np.nanpercentile(out.loc[out["hydraulic_solution_valid"], "H_base_m"], 95))],
        ["u_star_median_m_s", float(np.nanmedian(out.loc[out["hydraulic_solution_valid"], "u_star_base_m_s"]))],
    ], columns=["metric", "value"])
    return out, q


@dataclass
class ProfileCalibration:
    lambda_values: np.ndarray
    summary: pd.DataFrame
    status: str
    proxy_fraction: float

def _fit_profile_group(g: pd.DataFrame, cfg: dict):
    """Fit the effective particle settling–mixing scale from depth profiles."""
    z = pd.to_numeric(g["z_m"], errors="coerce").to_numpy(float)
    Hs = pd.to_numeric(g["H_m"], errors="coerce").to_numpy(float)
    C = pd.to_numeric(g["concentration"], errors="coerce").to_numpy(float)
    us = pd.to_numeric(g["u_star_m_s"], errors="coerce").to_numpy(float)

    m = np.isfinite(z)&np.isfinite(Hs)&np.isfinite(C)&np.isfinite(us)&(
        z>0)&(C>0)&(Hs>z)&(us>0)
    if m.sum() < cfg["profile_min_points"]:
        return None

    z = z[m]; Hs = Hs[m]; C = C[m]; us = us[m]
    alpha = cfg["ref_a_ratio"]

    rouse_coord = ((Hs-z)/z) * (alpha/(1-alpha))
    good = np.isfinite(rouse_coord) & (rouse_coord > 0)
    if good.sum() < cfg["profile_min_points"]:
        return None

    z= z[good]; Hs=Hs[good]; C=C[good]; us=us[good]
    rouse_coord=rouse_coord[good]

    T = np.log(rouse_coord) / (cfg["kappa"] * us)
    Y = np.log(C)
    if np.nanstd(T) < 1e-12:
        return None

    lam, intercept = np.polyfit(T, Y, 1)
    yhat = intercept + lam*T
    ssr = float(np.sum((Y-yhat)**2))
    sst = float(np.sum((Y-Y.mean())**2))
    r2 = 1-ssr/sst if sst>0 else np.nan

    H = float(np.nanmedian(Hs))
    ustar = float(np.nanmedian(us))
    P_equiv = lam/(cfg["kappa"]*ustar)

    if not (np.isfinite(lam) and lam>0 and np.isfinite(P_equiv) and P_equiv>0
            and np.isfinite(r2) and r2>=cfg["profile_min_r2"]):
        return None

    return P_equiv, r2, lam, H, ustar, int(len(Y))

def calibrate_profiles(path: str, cfg: dict) -> ProfileCalibration:
    if not path or not Path(path).exists():
        raise FileNotFoundError("A depth-resolved profile calibration file is required.")
    d=load_table(path)
    req={"profile_id","z_m","H_m","concentration","u_star_m_s"}
    miss=req-set(d.columns)
    if miss: raise ValueError(f"Profile file missing {sorted(miss)}")
    if "split" not in d.columns: d["split"]="calibration"
    if "dataset" not in d.columns: d["dataset"]="unknown"
    if "profile_kind" not in d.columns: d["profile_kind"]="unspecified"

    calrows=[]
    caldata=d[d["split"].astype(str).str.lower().eq("calibration")]
    for pid,g in caldata.groupby("profile_id"):
        fit=_fit_profile_group(g,cfg)
        if fit is None: continue
        Pfit,r2,lam,H,ustar,npts=fit
        kind=str(g["profile_kind"].iloc[0])
        calrows.append({"profile_id":pid,"dataset":g["dataset"].iloc[0],"profile_kind":kind,
                        "n_depths":npts,"H_m":H,"u_star_m_s":ustar,"P_fit":Pfit,
                        "R2_log_profile":r2,"lambda_PP_m_s":lam})
    s=pd.DataFrame(calrows)
    if s.empty:
        raise ValueError("No profile passed the calibration-quality criteria.")
    if len(s)<cfg["profile_min_accepted_final"]:
        raise ValueError(f"Only {len(s)} calibration profiles accepted; final minimum is {cfg['profile_min_accepted_final']}.")
    proxy = float((~s["profile_kind"].astype(str).str.upper().eq("PP")).mean())
    status=f"EMPIRICAL_N{len(s)}_NONPP_PROXY_FRACTION_{proxy:.2f}"
    return ProfileCalibration(s["lambda_PP_m_s"].to_numpy(float),s,status,proxy)


@dataclass
class FppPrior:
    values: np.ndarray
    status: str

def load_fpp_prior(path: str, cfg: dict) -> FppPrior:
    if not path or not Path(path).exists():
        return FppPrior(np.array([]), "SCENARIO_NO_EMPIRICAL_FPP")
    d=load_table(path)
    if "f_pp" in d.columns:
        f=pd.to_numeric(d["f_pp"],errors="coerce").to_numpy(float)
    elif {"TP","DP"}.issubset(d.columns):
        tp=pd.to_numeric(d["TP"],errors="coerce").to_numpy(float)
        dp=pd.to_numeric(d["DP"],errors="coerce").to_numpy(float)
        f=(tp-dp)/tp
    elif {"TP","PP"}.issubset(d.columns):
        tp=pd.to_numeric(d["TP"],errors="coerce").to_numpy(float)
        pp=pd.to_numeric(d["PP"],errors="coerce").to_numpy(float)
        f=pp/tp
    else:
        raise ValueError("f_PP file requires f_pp, or paired TP+DP, or TP+PP columns.")
    f=f[np.isfinite(f)&(f>=0)&(f<=1)]
    if len(f)==0: raise ValueError("No valid f_PP values found.")
    q=np.percentile(f,[5,25,50,75,95])
    return FppPrior(f, f"EMPIRICAL_FPP_N{len(f)}_P05_{q[0]:.3f}_MEDIAN_{q[2]:.3f}_P95_{q[4]:.3f}")


def build_kpp_lookup(cfg: dict):
    alpha=cfg["ref_a_ratio"]; eps=cfg["z0_ratio"]
    if not (0<eps<=alpha<1): raise ValueError("Require 0<z0_ratio<=ref_a_ratio<1")
    Pgrid=np.geomspace(cfg["P_min"],cfg["P_support_max"],cfg["P_grid_size"])
    elo,ehi=cfg["sample_height_ratio_range"]
    etagrid=np.linspace(elo,ehi,cfg["eta_grid_size"])
    xi=np.linspace(max(alpha,eps*(1+1e-9)),1-1e-6,cfg["vertical_grid_size"])
    v=np.log(xi/eps); vden=np.trapezoid(v,xi)
    base=((1-xi)/xi)*(alpha/(1-alpha)); logbase=np.log(np.maximum(base,1e-300))
    ebase=((1-etagrid)/etagrid)*(alpha/(1-alpha)); ebase=np.maximum(ebase,1e-300)
    K=np.empty((len(Pgrid),len(etagrid)))
    for i,P in enumerate(Pgrid):
        c=np.exp(P*logbase)
        cbar=np.trapezoid(c*v,xi)/vden
        K[i,:]=cbar/(ebase**P)-1
    interp=RegularGridInterpolator((np.log(Pgrid),etagrid),K,bounds_error=False,fill_value=np.nan)
    return Pgrid,etagrid,interp

def calculate_corrections(h: pd.DataFrame, cal: ProfileCalibration, fpp: FppPrior, cfg: dict):
    _,_,lookup=build_kpp_lookup(cfg)
    rng=np.random.default_rng(cfg["random_seed"]); M=cfg["ensemble_size"]

    lam_draw=rng.choice(cal.lambda_values,size=M,replace=True)
    lam0=float(np.nanmedian(cal.lambda_values))

    elo,ehi=cfg["sample_height_ratio_range"]
    eta_draw=rng.uniform(elo,ehi,size=M); eta0=cfg["sample_height_ratio_baseline"]

    if fpp.values.size:
        f_draw=rng.choice(fpp.values,size=M,replace=True)
        f0=float(np.nanmedian(fpp.values))
    else:
        flo,fhi=cfg["f_pp_scenario_range"]
        f_draw=rng.uniform(flo,fhi,size=M); f0=cfg["f_pp_baseline"]

    rdraw=rng.uniform(0,1,size=M)
    parts=[]
    for st in range(0,len(h),cfg["chunk_size"]):
        x=h.iloc[st:st+cfg["chunk_size"]].copy()
        u0=x["u_star_base_m_s"].to_numpy(float)
        ulo=np.minimum(x["u_star_nlo_m_s"],x["u_star_nhi_m_s"]).to_numpy(float)
        uhi=np.maximum(x["u_star_nlo_m_s"],x["u_star_nhi_m_s"]).to_numpy(float)
        valid=x["hydraulic_solution_valid"].to_numpy(bool)

        ld=np.log(np.maximum(ulo,1e-12))[:,None]; hd=np.log(np.maximum(uhi,1e-12))[:,None]
        udraw=np.exp(ld+(hd-ld)*rdraw[None,:])
        P=lam_draw[None,:]/(cfg["kappa"]*udraw)

        # Out-of-domain P draws are excluded, never clipped into the fit domain.
        supported=(
            valid[:,None]
            & np.isfinite(P)
            & (P>=cfg["P_min"])
            & (P<=cfg["P_support_max"])
        )

        pts=np.column_stack([
            np.log(np.clip(P,cfg["P_min"],cfg["P_support_max"])).ravel(),
            np.broadcast_to(eta_draw,P.shape).ravel()
        ])
        Kpp=lookup(pts).reshape(P.shape)
        Kpp[~supported]=np.nan
        Ktp=Kpp*f_draw[None,:]

        supported_fraction=np.mean(supported,axis=1)
        n_supported=np.sum(supported,axis=1)
        # Primary results require >=80% supported ensemble draws.
        applicable=(
            valid
            & (supported_fraction>=cfg["minimum_supported_ensemble_fraction"])
        )

        P0=lam0/(cfg["kappa"]*np.maximum(u0,1e-12))
        baseline_supported=(
            valid
            & np.isfinite(P0)
            & (P0>=cfg["P_min"])
            & (P0<=cfg["P_support_max"])
        )
        pts0=np.column_stack([
            np.log(np.clip(P0,cfg["P_min"],cfg["P_support_max"])),
            np.full(len(x),eta0)
        ])
        Kpp0=lookup(pts0)
        Kpp0[~baseline_supported]=np.nan
        Ktp0=Kpp0*f0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore",RuntimeWarning)
            out=pd.DataFrame(index=x.index)
            out["Rouse_P_baseline"]=P0
            out["baseline_within_empirical_P_domain"]=baseline_supported
            out["K_pp_baseline"]=Kpp0
            out["K_tp_baseline"]=Ktp0

            for name,arr in (
                ("Rouse_P",np.where(supported,P,np.nan)),
                ("K_pp",Kpp),
                ("K_tp",Ktp)
            ):
                q05=np.nanpercentile(arr,5,axis=1)
                q50=np.nanpercentile(arr,50,axis=1)
                q95=np.nanpercentile(arr,95,axis=1)

                out[f"{name}_conditional_p05"]=q05
                out[f"{name}_conditional_median"]=q50
                out[f"{name}_conditional_p95"]=q95

                out[f"{name}_p05"]=np.where(applicable,q05,np.nan)
                out[f"{name}_median"]=np.where(applicable,q50,np.nan)
                out[f"{name}_p95"]=np.where(applicable,q95,np.nan)

            out["supported_ensemble_fraction"]=supported_fraction
            out["out_of_domain_ensemble_fraction"]=1.0-supported_fraction
            out["n_supported_ensemble_draws"]=n_supported
            out["model_applicable_primary"]=applicable
            out["valid_ensemble_fraction"]=supported_fraction
            out["P_support_max_used"]=cfg["P_support_max"]
            out["minimum_supported_ensemble_fraction_used"]=cfg["minimum_supported_ensemble_fraction"]

            out["lambda_PP_baseline_m_s"]=lam0
            out["sample_height_baseline_z_over_H"]=eta0
            out["f_pp_baseline_used"]=f0
            out["lambda_status"]=cal.status
            out["f_pp_status"]=fpp.status
        parts.append(pd.concat([x,out],axis=1))
    return pd.concat(parts).reset_index(drop=True)


def validate_profile_holdout(profile_path: str, cal: ProfileCalibration, cfg: dict):
    """Evaluate independent profiles using median predictions and interval coverage."""
    if not profile_path or not Path(profile_path).exists() or cal.lambda_values.size == 0:
        return pd.DataFrame(), pd.DataFrame()

    d = load_table(profile_path)
    if "split" not in d.columns:
        return pd.DataFrame(), pd.DataFrame()

    v = d[d["split"].astype(str).str.lower().eq("validation")]
    if v.empty:
        return pd.DataFrame(), pd.DataFrame()

    lam = np.asarray(cal.lambda_values, dtype=float)
    lam = lam[np.isfinite(lam) & (lam > 0)]
    if lam.size == 0:
        return pd.DataFrame(), pd.DataFrame()

    lam_q05, lam_q50, lam_q95 = np.nanpercentile(lam, [5, 50, 95])

    rows = []
    for pid, g in v.groupby("profile_id"):
        fit = _fit_profile_group(g, cfg)
        if fit is None:
            continue

        Pobs, r2, lamobs, H, ustar, npts = fit

        p_draw = lam / (cfg["kappa"] * ustar)
        p05, p50, p95 = np.nanpercentile(p_draw, [5, 50, 95])

        rows.append({
            "profile_id": pid,
            "dataset": g["dataset"].iloc[0] if "dataset" in g else "unknown",
            "P_observed": Pobs,
            "P_predicted_p05": p05,
            "P_predicted_median": p50,
            "P_predicted_p95": p95,
            "P_covered_by_90pct_interval": bool((Pobs >= p05) and (Pobs <= p95)),
            "lambda_observed_m_s": lamobs,
            "lambda_calibration_p05_m_s": lam_q05,
            "lambda_calibration_median_m_s": lam_q50,
            "lambda_calibration_p95_m_s": lam_q95,
            "lambda_covered_by_90pct_interval": bool((lamobs >= lam_q05) and (lamobs <= lam_q95)),
            "R2_profile_fit": r2,
            "u_star_m_s": ustar,
            "H_m": H,
            "n_depths": npts,
        })

    r = pd.DataFrame(rows)
    if r.empty:
        return r, pd.DataFrame()

    err = r["P_predicted_median"] - r["P_observed"]
    corr = (
        float(r[["P_observed", "P_predicted_median"]].corr().iloc[0, 1])
        if len(r) > 1 else np.nan
    )

    lam_obs_med = float(np.nanmedian(r["lambda_observed_m_s"]))
    metrics = pd.DataFrame([
        ["n_profiles", len(r)],
        ["P_RMSE_median_prediction", float(np.sqrt(np.mean(err**2)))],
        ["P_bias_median_prediction", float(np.mean(err))],
        ["P_MAE_median_prediction", float(np.mean(np.abs(err)))],
        ["P_corr_median_prediction", corr],
        ["P_90pct_interval_coverage",
         float(np.mean(r["P_covered_by_90pct_interval"].astype(float)))],
        ["lambda_90pct_interval_coverage",
         float(np.mean(r["lambda_covered_by_90pct_interval"].astype(float)))],
        ["lambda_calibration_median_m_s", float(lam_q50)],
        ["lambda_validation_median_m_s", lam_obs_med],
        ["lambda_validation_to_calibration_median_ratio",
         float(lam_obs_med / lam_q50) if lam_q50 > 0 else np.nan],
    ], columns=["metric", "value"])

    return r, metrics

def validate_hydraulics(global_df: pd.DataFrame, benchmark_path: str):
    if not benchmark_path or not Path(benchmark_path).exists(): return pd.DataFrame(),pd.DataFrame()
    b=load_table(benchmark_path)
    req={"reach_id","depth_meas_m"}; miss=req-set(b.columns)
    if miss: raise ValueError(f"Hydraulic benchmark missing {sorted(miss)}")
    b["reach_id"]=normalize_reach_id(b["reach_id"]); b["depth_meas_m"]=pd.to_numeric(b["depth_meas_m"],errors="coerce")
    pred=global_df[["reach_id","H_base_m","hydraulic_solution_valid"]].drop_duplicates("reach_id")
    r=b.merge(pred,on="reach_id",how="inner"); r=r[np.isfinite(r["depth_meas_m"])&(r["depth_meas_m"]>0)&r["hydraulic_solution_valid"]]
    if r.empty: return r,pd.DataFrame()
    e=r["H_base_m"]-r["depth_meas_m"]
    m=pd.DataFrame([["n_measurements",len(r)],["depth_RMSE_m",float(np.sqrt(np.mean(e**2)))],
                    ["depth_MAE_m",float(np.mean(np.abs(e)))],["depth_bias_m",float(np.mean(e))],
                    ["depth_median_ratio_pred_obs",float(np.median(r["H_base_m"]/r["depth_meas_m"]))]],columns=["metric","value"])
    return r,m

def validate_tp_pairs(path: str):
    if not path or not Path(path).exists(): return pd.DataFrame(),pd.DataFrame()
    d=load_table(path)
    req={"TP_single","TP_integrated"}; miss=req-set(d.columns)
    if miss: raise ValueError(f"TP pair benchmark missing {sorted(miss)}")
    a=pd.to_numeric(d["TP_single"],errors="coerce"); b=pd.to_numeric(d["TP_integrated"],errors="coerce")
    d=d[np.isfinite(a)&np.isfinite(b)&(a>0)].copy(); d["K_tp_observed"]=pd.to_numeric(d["TP_integrated"],errors="coerce")/pd.to_numeric(d["TP_single"],errors="coerce")-1
    if d.empty: return d,pd.DataFrame()
    q=np.percentile(d["K_tp_observed"],[5,25,50,75,95])
    m=pd.DataFrame([["n_pairs",len(d)],["Ktp_obs_p05",q[0]],["Ktp_obs_p25",q[1]],["Ktp_obs_median",q[2]],["Ktp_obs_p75",q[3]],["Ktp_obs_p95",q[4]]],columns=["metric","value"])
    return d,m


def parameter_table(cfg: dict, user: dict, cal: ProfileCalibration, fpp: FppPrior):
    rows=[]
    for k,v in cfg.items(): rows.append([k,str(v)])
    rows += [["SWORD_version","v17b"],["profile_calibration_status",cal.status],["f_pp_status",fpp.status],
             ["FINAL_MODE",str(user["FINAL_MODE"])]]
    return pd.DataFrame(rows,columns=["parameter","value"])


def main():
    cfg=PHYSICS
    args=parse_args()
    user={
        "sword_source_file": args.sword,
        "discharge_prior_file": args.discharge,
        "sos_prior_dir": args.sos_prior_dir,
        "profile_file": args.profiles,
        "fpp_file": args.fpp,
        "hydraulic_benchmark_file": args.hydraulic_benchmark,
        "tp_pair_benchmark_file": args.tp_benchmark,
        "output_dir": args.outdir,
        "max_rows": args.max_rows,
        "FINAL_MODE": True,
    }
    outdir=Path(user["output_dir"]); outdir.mkdir(parents=True,exist_ok=True)

    print("=== CEE revised K_TP pipeline v5.4 | empirical applicability domain ===")
    src=load_table(user["sword_source_file"], user.get("max_rows"))
    src,qa_source=validate_sword_source(src,cfg)
    print(f"SWORD source rows: {len(src):,}")

    qjoined,qa_q,qa_q_files=join_discharge_prior(src,user["discharge_prior_file"],user["sos_prior_dir"],cfg)
    hydro,qa_h=reconstruct_hydraulics(qjoined,cfg)

    cal=calibrate_profiles(user["profile_file"],cfg)
    fpp=load_fpp_prior(user.get("fpp_file",""),cfg)
    print("Profile calibration:",cal.status)
    print("f_PP:",fpp.status)

    result=calculate_corrections(hydro,cal,fpp,cfg)
    result_path=outdir/"global_Ktp_revised.xlsx"
    result.to_excel(result_path,index=False)

    prof_val,prof_metrics=validate_profile_holdout(user.get("profile_file",""),cal,cfg)
    hyd_val,hyd_metrics=validate_hydraulics(result,user.get("hydraulic_benchmark_file",""))
    tp_val,tp_metrics=validate_tp_pairs(user.get("tp_pair_benchmark_file",""))

    applicable_mask=result["model_applicable_primary"].fillna(False).astype(bool)
    finite=result.loc[applicable_mask,"K_tp_median"].replace([np.inf,-np.inf],np.nan).dropna()
    if len(finite):
        kq=np.percentile(finite,[5,25,50,75,95])
    else:
        kq=[np.nan]*5

    sf=result.loc[result["hydraulic_solution_valid"],"supported_ensemble_fraction"].replace(
        [np.inf,-np.inf],np.nan
    ).dropna()
    if len(sf):
        sfq=np.percentile(sf,[5,50,95])
    else:
        sfq=[np.nan]*3

    n_hyd=int(result["hydraulic_solution_valid"].sum())
    n_app=int(applicable_mask.sum())

    qa_results=pd.DataFrame([
        ["source_rows",len(result)],
        ["hydraulic_solution_valid",n_hyd],
        ["P_support_max_primary",cfg["P_support_max"]],
        ["minimum_supported_ensemble_fraction_primary",cfg["minimum_supported_ensemble_fraction"]],
        ["model_applicable_primary_reaches",n_app],
        ["model_applicable_fraction_of_hydraulic_reaches",(n_app/n_hyd if n_hyd else np.nan)],
        ["supported_ensemble_fraction_p05_hydraulic_reaches",sfq[0]],
        ["supported_ensemble_fraction_median_hydraulic_reaches",sfq[1]],
        ["supported_ensemble_fraction_p95_hydraulic_reaches",sfq[2]],
        ["finite_primary_Ktp_median",len(finite)],
        ["Ktp_p05_across_primary_reaches",kq[0]],
        ["Ktp_p25_across_primary_reaches",kq[1]],
        ["Ktp_median_across_primary_reaches",kq[2]],
        ["Ktp_p75_across_primary_reaches",kq[3]],
        ["Ktp_p95_across_primary_reaches",kq[4]],
        ["negative_primary_Ktp_median_count",int((finite<0).sum())],
        ["lambda_status",cal.status],
        ["f_pp_status",fpp.status],
    ],columns=["metric","value"])

    app_rows=[]
    for thr in (0.50,0.80,0.90,0.95):
        m=(
            result["hydraulic_solution_valid"].fillna(False).astype(bool)
            & (result["supported_ensemble_fraction"]>=thr)
        )
        vals=result.loc[m,"K_tp_conditional_median"].replace([np.inf,-np.inf],np.nan).dropna()
        if len(vals):
            q=np.percentile(vals,[5,25,50,75,95])
        else:
            q=[np.nan]*5
        app_rows.append([
            thr,int(m.sum()),(float(m.sum())/n_hyd if n_hyd else np.nan),
            q[0],q[1],q[2],q[3],q[4]
        ])
    qa_applicability=pd.DataFrame(
        app_rows,
        columns=[
            "minimum_supported_fraction","n_reaches","fraction_of_hydraulic_reaches",
            "Ktp_p05","Ktp_p25","Ktp_median","Ktp_p75","Ktp_p95"
        ]
    )

    qa_path=outdir/"QA_summary.xlsx"
    with pd.ExcelWriter(qa_path,engine="openpyxl") as w:
        qa_source.to_excel(w, sheet_name="SWORD_source_QA",index=False)
        qa_q.to_excel(w, sheet_name="Discharge_match_QA",index=False)
        if not qa_q_files.empty: qa_q_files.to_excel(w, sheet_name="SoS_prior_files_QA",index=False)
        qa_h.to_excel(w, sheet_name="Hydraulic_QA",index=False)
        qa_results.to_excel(w, sheet_name="Result_QA",index=False)
        qa_applicability.to_excel(w, sheet_name="Applicability_QA",index=False)
        parameter_table(cfg,user,cal,fpp).to_excel(w, sheet_name="Parameters",index=False)
        if not cal.summary.empty: cal.summary.to_excel(w, sheet_name="Profile_calibration",index=False)
        if not prof_metrics.empty: prof_metrics.to_excel(w, sheet_name="Profile_validation_metrics",index=False)
        if not hyd_metrics.empty: hyd_metrics.to_excel(w, sheet_name="Hydraulic_validation_metrics",index=False)
        if not tp_metrics.empty: tp_metrics.to_excel(w, sheet_name="TP_pair_metrics",index=False)

    if not cal.summary.empty: cal.summary.to_excel(outdir/"profile_calibration.xlsx",index=False)
    if any(not x.empty for x in (prof_val,hyd_val,tp_val)):
        with pd.ExcelWriter(outdir/"validation_details.xlsx",engine="openpyxl") as w:
            if not prof_val.empty: prof_val.to_excel(w, sheet_name="Profile_holdout",index=False)
            if not hyd_val.empty: hyd_val.to_excel(w, sheet_name="Hydraulic_benchmark",index=False)
            if not tp_val.empty: tp_val.to_excel(w, sheet_name="TP_pairs",index=False)

    print("Saved:",result_path)
    print("Saved:",qa_path)
    print("Primary valid slope count:",int(src["slope_valid"].sum()),"/",len(src))
    print("Discharge match rate:",qa_q.loc[qa_q.metric.eq("match_rate"),"value"].iloc[0] if (qa_q.metric=="match_rate").any() else "NA")
    print("Primary applicable reaches:",n_app,"/",n_hyd)
    print("Primary K_TP across-reach 5/25/50/75/95%:",np.round(kq,4))


if __name__ == "__main__":
    main()
