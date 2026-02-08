import pandas as pd
import numpy as np
from scipy.integrate import quad, IntegrationWarning
import warnings
import os

# --- Physics Constants and Configuration ---
PHYSICS_CONFIG = {
    'g': 9.81,              # Gravitational acceleration (m/s^2)
    'kappa': 0.41,          # von Karman constant
    'P_range': [0.05, 0.1, 0.2, 0.4, 0.8], # Range of Rouse numbers (P)
    'z0_ratio': 0.001,      # Ratio of roughness length to water depth
    'ref_a_ratio': 0.05,    # Ratio of reference height to water depth
    'use_slope_clipping': True,
    'slope_min_m_per_km': 0.01,
    'slope_max_m_per_km': 100.0,
    'u_star_min': 1e-6,     # Minimum shear velocity threshold (m/s)
    'default_pp_fraction': 0.7
}

# LULC to Particulate Phosphorus Fraction
LULC_PP_MAP = {
    'Cropland': 0.85, 'Forest': 0.50, 'Grassland': 0.60,
    'Shrubland': 0.60, 'Tundra': 0.60, 'Barren land': 0.90,
    'Snow/Ice': 0.40, 'No data': 0.70
}

USER_CONFIG = {
    'river_data_file': "path1", # Input data file path
    'output_file': "path2" # Output file path
}

# --- Data Loading Module ---
def load_river_data(filepath):
    print(f"--- Loading river data: {filepath} ---")
    if not os.path.exists(filepath):
        print("Error")
        return None
    try:
        df = pd.read_excel(filepath, engine='openpyxl')
    except Exception as e:
        print(f"Failed to read Excel: {e}")
        return None
        
    required_cols = ['reach_id', 'width', 'facc', 'slope', 'runoff', 'n', 'lulc']
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"Error: Missing columns in data: {missing_cols}")
        return None

    df.dropna(subset=required_cols, inplace=True)
    
    for col in ['width', 'slope', 'facc', 'runoff', 'n']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df[(df['width'] > 0) & (df['slope'] > 0)]
    df['reach_id'] = df['reach_id'].astype(str)
    
    print(f"  -> 有效计算数据行数: {len(df)}")
    return df

# --- Physics Calculation Module ---
def calculate_correction_factor(river_data):
    """
    Core Algorithm: Calculate the K_tp correction factor for each river reach.
    1. Hydraulic parameter estimation (Depth H, Shear Velocity u_star).
    2. Vertical concentration profile integration.
    """
    results_list = []
    total_rows = len(river_data)
    print(f"\n--- Starting calculation for {total_rows} reaches ---")
    
    warnings.filterwarnings('ignore', category=IntegrationWarning)
    cfg = PHYSICS_CONFIG

    for index, reach in river_data.iterrows():
        if (index + 1) % 100 == 0: 
            print(f"  -> Progress: {index + 1}/{total_rows}", end='\r')
        
        try:
            # 1. Extract Basic Hydraulic Parameters
            width = float(reach['width'])
            n = float(reach['n']) # Manning's roughness coefficient
            
            # Slope 
            slope_m = float(np.clip(reach['slope'], cfg['slope_min_m_per_km'], cfg['slope_max_m_per_km'])) / 1000.0
            
            # Discharge calculation
            Q = float(reach['facc']) * 1e6 * float(reach['runoff']) / 86400.0
            
            if Q <= 1e-4: continue 

            # 2. Calculate Depth (H) and Shear Velocity (u_star)
            H = ((Q * n) / (width * np.sqrt(slope_m))) ** 0.6
            u_star = np.sqrt(cfg['g'] * H * slope_m)
            if u_star <= cfg.get('u_star_min', 1e-6): continue

            # 3. Get Particulate Phosphorus (PP) Fraction
            lulc = str(reach['lulc']).strip()
            pp_frac = float(LULC_PP_MAP.get(lulc, cfg['default_pp_fraction']))
            
            # 4. Define Integration Boundaries
            z0 = max(H * cfg['z0_ratio'], 1e-5)      # Hydrodynamic roughness length
            a = max(H * cfg['ref_a_ratio'], z0 * 1.1) # Reference height
            
            K_tp_vals = []
            
            # Velocity Profile Function
            def u_func(z): 
                return (u_star / cfg['kappa']) * np.log(z / z0) if z > z0 else 0.0

            # 5. Integrate for different Rouse numbers (P)
            for P in cfg['P_range']:
                # Sediment Concentration Profile (Rouse Profile)
                def C_func(z):
                    if z <= a: return 1.0
                    z_eff = min(z, H - 1e-6) 
                    term = ((H - z_eff) / z_eff) * (a / (H - a))
                    return 1.0 * (term ** float(P)) if term > 0 else 0.0
                
                # Calculate Actual Flux
                flux_acc, _ = quad(lambda z: C_func(z) * u_func(z) * width, a, H)
                
                # Calculate Simplified Flux
                flux_simple = C_func(0.6 * H) * Q 
                
                if flux_simple > 1e-12:
                    # Calculate Correction Factor K
                    # K = (Actual Flux / Simplified Flux - 1) * Particulate Fraction
                    K_tp = ((flux_acc / flux_simple) - 1.0) * pp_frac
                    if np.isfinite(K_tp): 
                        K_tp_vals.append(K_tp)

            if not K_tp_vals: continue
            
            p05, p50, p95 = np.percentile(K_tp_vals, [5, 50, 95])
            
            results_list.append({
                'reach_id': str(reach['reach_id']),
                'x': reach.get('x', np.nan), 
                'y': reach.get('y', np.nan),
                'Q_m3s': Q, 
                'H_m': H, 
                'u_star': u_star, 
                'slope_km': slope_m * 1000,
                'lulc': lulc, 
                'pp_fraction': pp_frac,
                'K_tp_median': p50, 
                'K_tp_p05': p05, 
                'K_tp_p95': p95
            })
        except Exception as e:
            continue

    print("\n...Calculation completed!")
    return pd.DataFrame(results_list)

# --- Main Execution ---
if __name__ == "__main__":
    df_river = load_river_data(USER_CONFIG['river_data_file'])
    
    if df_river is not None:
        df_res = calculate_correction_factor(df_river)
        if not df_res.empty:
            output_path = USER_CONFIG['output_file']
            df_res.to_excel(output_path, index=False)
            print(f"Success: Results saved to {output_path}")
            print(f"Columns included: {list(df_res.columns)}")
        else:
            print("No valid calculation results were generated")