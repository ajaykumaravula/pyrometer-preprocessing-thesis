"""
=============================================================================
calibrate.py  —  D2: ATP-2 Calibration Module
=============================================================================
Thesis  : Automation of pyrometer data pre-processing
          (ATP-2 Calibration + ATP-3 Compression)
Author  : [Your Name]

WHAT THIS MODULE DOES:
  Provides 8 calibration methods that correct systematic pyrometer
  temperature errors using a thermocouple reference signal.
  Each function takes a denoised pyrometer signal + thermocouple
  reference and returns a calibrated signal.

ATP-2 CALIBRATION METHODS (Research Question 1):
  Classical:
    1. mean_offset          — subtract average offset over cal window
    2. linear               — fit T_ref = a*T_pyr + b
    3. polynomial           — fit degree-2 polynomial
    4. piecewise_linear     — 3-segment linear fit
  ML/AI:
    5. random_forest        — 100-tree ensemble regressor
    6. mlp                  — neural network (3→64→32→1)
    7. gradient_boosting    — sequential boosted trees
    8. svr                  — support vector regression (RBF kernel)

HOW TO USE:
  from calibrate import calibrate, METHODS, rmse, mae

  # Calibrate with one method
  T_cal, coeffs = calibrate(T_pyr, T_tc, method='linear')

  # Run all 8 methods and compare
  for name in METHODS:
      T_cal, _ = calibrate(T_pyr, T_tc, method=name)

SWAP DATA:
  All functions accept numpy arrays — works with both simulated
  and real AP&T 2-pyrometer + thermocouple data unchanged.
=============================================================================
"""

import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Registry of all available method names
METHODS = [
    'mean_offset',
    'linear',
    'polynomial',
    'piecewise_linear',
    'random_forest',
    'mlp',
    'gradient_boosting',
    'svr',
]


# =============================================================================
# ACCURACY METRICS
# =============================================================================

def rmse(T_cal: np.ndarray, T_ref: np.ndarray) -> float:
    """
    Root Mean Square Error between calibrated and reference signal.

    Parameters
    ----------
    T_cal : calibrated pyrometer signal (°C)
    T_ref : thermocouple reference signal (°C)

    Returns
    -------
    float — RMSE in °C
    """
    return float(np.sqrt(np.mean((T_cal - T_ref) ** 2)))


def mae(T_cal: np.ndarray, T_ref: np.ndarray) -> float:
    """
    Mean Absolute Error between calibrated and reference signal.

    Parameters
    ----------
    T_cal : calibrated pyrometer signal (°C)
    T_ref : thermocouple reference signal (°C)

    Returns
    -------
    float — MAE in °C
    """
    return float(np.mean(np.abs(T_cal - T_ref)))


def max_error(T_cal: np.ndarray, T_ref: np.ndarray) -> float:
    """
    Maximum absolute error between calibrated and reference signal.

    Parameters
    ----------
    T_cal : calibrated pyrometer signal (°C)
    T_ref : thermocouple reference signal (°C)

    Returns
    -------
    float — Max error in °C
    """
    return float(np.max(np.abs(T_cal - T_ref)))


# =============================================================================
# HELPER — FEATURE ENGINEERING
# =============================================================================

def _make_features(T: np.ndarray) -> np.ndarray:
    """
    Build feature matrix for ML methods.
    Features: [T, T², √|T|] — captures nonlinear emissivity behaviour.

    Parameters
    ----------
    T : 1-D pyrometer signal array

    Returns
    -------
    np.ndarray — shape (n, 3)
    """
    T = T.astype(np.float64)
    return np.column_stack([T, T ** 2, np.sqrt(np.abs(T))])


# =============================================================================
# CLASSICAL METHOD 1 — MEAN OFFSET
# =============================================================================

def mean_offset(T_pyr: np.ndarray,
                T_ref: np.ndarray,
                cal_fraction: float = 0.20) -> tuple:
    """
    Mean Offset Calibration.

    Computes the mean difference between pyrometer and thermocouple
    over the calibration window, then subtracts it from the full signal.

        offset = mean(T_pyr[:cal_end] - T_ref[:cal_end])
        T_cal  = T_pyr - offset

    Suitable for: constant emissivity error (systematic bias).

    Parameters
    ----------
    T_pyr        : denoised pyrometer signal (°C)
    T_ref        : thermocouple reference signal (°C)
    cal_fraction : fraction of signal used as calibration window

    Returns
    -------
    T_cal  : np.ndarray — calibrated signal
    coeffs : dict       — offset value and calibration window size
    """
    cal_end = max(10, int(cal_fraction * len(T_pyr)))
    offset  = float(np.mean(
        T_pyr[:cal_end].astype(np.float64) -
        T_ref[:cal_end].astype(np.float64)
    ))
    T_cal = T_pyr.astype(np.float64) - offset
    return T_cal, {
        'method'  : 'Mean Offset',
        'offset'  : offset,
        'cal_end' : cal_end,
    }


# =============================================================================
# CLASSICAL METHOD 2 — LINEAR REGRESSION
# =============================================================================

def linear(T_pyr: np.ndarray,
           T_ref: np.ndarray,
           cal_fraction: float = 0.20) -> tuple:
    """
    Linear Regression Calibration.

    Fits T_ref ≈ a × T_pyr + b on the calibration window using
    ordinary least squares, then applies the correction to all data.

        T_cal = a × T_pyr + b

    Suitable for: emissivity that is constant but scale-dependent.

    Parameters
    ----------
    T_pyr        : denoised pyrometer signal (°C)
    T_ref        : thermocouple reference signal (°C)
    cal_fraction : fraction of signal used as calibration window

    Returns
    -------
    T_cal  : np.ndarray — calibrated signal
    coeffs : dict       — slope a, intercept b
    """
    cal_end = max(10, int(cal_fraction * len(T_pyr)))
    x = T_pyr[:cal_end].astype(np.float64)
    y = T_ref[:cal_end].astype(np.float64)

    # Least-squares: stack [x | 1] and solve for [a, b]
    A    = np.vstack([x, np.ones(len(x))]).T
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    T_cal = a * T_pyr.astype(np.float64) + b

    return T_cal, {
        'method'  : 'Linear',
        'a'       : a,
        'b'       : b,
        'cal_end' : cal_end,
    }


# =============================================================================
# CLASSICAL METHOD 3 — POLYNOMIAL REGRESSION
# =============================================================================

def polynomial(T_pyr: np.ndarray,
               T_ref: np.ndarray,
               cal_fraction: float = 0.20,
               degree: int = 2) -> tuple:
    """
    Polynomial Regression Calibration (degree = 2).

    Fits a degree-2 polynomial T_ref ≈ p(T_pyr) on the calibration
    window. Handles emissivity that changes nonlinearly with temperature.

    Parameters
    ----------
    T_pyr        : denoised pyrometer signal (°C)
    T_ref        : thermocouple reference signal (°C)
    cal_fraction : fraction of signal used as calibration window
    degree       : polynomial degree (default 2)

    Returns
    -------
    T_cal  : np.ndarray — calibrated signal
    coeffs : dict       — polynomial coefficients and degree
    """
    cal_end  = max(10, int(cal_fraction * len(T_pyr)))
    x        = T_pyr[:cal_end].astype(np.float64)
    y        = T_ref[:cal_end].astype(np.float64)
    poly_c   = np.polyfit(x, y, deg=degree)
    poly_fn  = np.poly1d(poly_c)
    T_cal    = poly_fn(T_pyr.astype(np.float64))

    return T_cal, {
        'method'      : f'Polynomial(deg={degree})',
        'poly_coeffs' : poly_c,
        'degree'      : degree,
        'cal_end'     : cal_end,
    }


# =============================================================================
# CLASSICAL METHOD 4 — PIECEWISE LINEAR
# =============================================================================

def piecewise_linear(T_pyr: np.ndarray,
                     T_ref: np.ndarray,
                     cal_fraction: float = 0.20,
                     n_segments: int = 3) -> tuple:
    """
    Piecewise Linear Calibration.

    Divides the temperature range into n_segments and fits a separate
    linear regression in each segment. More flexible than a single
    linear fit — useful when emissivity behaves differently at low,
    mid and high temperatures (e.g. phase transitions in metal forming).

    Parameters
    ----------
    T_pyr        : denoised pyrometer signal (°C)
    T_ref        : thermocouple reference signal (°C)
    cal_fraction : fraction of signal used as calibration window
    n_segments   : number of temperature segments (default 3)

    Returns
    -------
    T_cal  : np.ndarray — calibrated signal
    coeffs : dict       — per-segment slopes, intercepts, breakpoints
    """
    cal_end = max(10, int(cal_fraction * len(T_pyr)))
    x_cal   = T_pyr[:cal_end].astype(np.float64)
    y_cal   = T_ref[:cal_end].astype(np.float64)

    # Equal-width breakpoints across the calibration temperature range
    T_min, T_max = x_cal.min(), x_cal.max()
    breakpoints  = np.linspace(T_min, T_max, n_segments + 1)

    # Fit one linear model per segment
    seg_coeffs = []
    for i in range(n_segments):
        lo, hi = breakpoints[i], breakpoints[i + 1]
        idx    = np.where((x_cal >= lo) & (x_cal <= hi))[0]
        if len(idx) < 2:
            # Not enough points — fall back to global linear fit
            A_all  = np.vstack([x_cal, np.ones(len(x_cal))]).T
            a, b   = np.linalg.lstsq(A_all, y_cal, rcond=None)[0]
        else:
            A  = np.vstack([x_cal[idx], np.ones(len(idx))]).T
            a, b = np.linalg.lstsq(A, y_cal[idx], rcond=None)[0]
        seg_coeffs.append((a, b, lo, hi))

    # Apply piecewise correction to the full signal
    T_full = T_pyr.astype(np.float64)
    T_cal  = np.empty_like(T_full)
    for a, b, lo, hi in seg_coeffs:
        if hi == breakpoints[-1]:
            mask = T_full >= lo          # last segment includes upper edge
        else:
            mask = (T_full >= lo) & (T_full < hi)
        T_cal[mask] = a * T_full[mask] + b

    # Handle values below the first breakpoint
    below = T_full < breakpoints[0]
    if below.any():
        a0, b0 = seg_coeffs[0][0], seg_coeffs[0][1]
        T_cal[below] = a0 * T_full[below] + b0

    return T_cal, {
        'method'      : 'Piecewise Linear',
        'n_segments'  : n_segments,
        'breakpoints' : breakpoints,
        'seg_coeffs'  : seg_coeffs,
        'cal_end'     : cal_end,
    }


# =============================================================================
# ML METHOD 5 — RANDOM FOREST
# =============================================================================

def random_forest(T_pyr: np.ndarray,
                  T_ref: np.ndarray,
                  cal_fraction: float = 0.20) -> tuple:
    """
    Random Forest Regression Calibration.

    Trains a 100-tree Random Forest on [T, T², √T] features extracted
    from the calibration window. Each tree independently learns the
    pyrometer-to-true-temperature mapping; predictions are averaged.

    Advantages: robust to outliers, no feature scaling needed,
    captures complex nonlinear emissivity behaviour.

    Parameters
    ----------
    T_pyr        : denoised pyrometer signal (°C)
    T_ref        : thermocouple reference signal (°C)
    cal_fraction : fraction of signal used for training

    Returns
    -------
    T_cal  : np.ndarray — calibrated signal
    coeffs : dict       — trained model
    """
    cal_end = max(10, int(cal_fraction * len(T_pyr)))
    X_train = _make_features(T_pyr[:cal_end])
    y_train = T_ref[:cal_end].astype(np.float64)

    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    T_cal = rf.predict(_make_features(T_pyr))

    return T_cal, {
        'method'       : 'Random Forest',
        'model'        : rf,
        'n_estimators' : 100,
        'cal_end'      : cal_end,
    }


# =============================================================================
# ML METHOD 6 — MLP NEURAL NETWORK
# =============================================================================

def mlp(T_pyr: np.ndarray,
        T_ref: np.ndarray,
        cal_fraction: float = 0.20) -> tuple:
    """
    MLP Neural Network Calibration.

    Architecture: input(3) → Dense(64, ReLU) → Dense(32, ReLU) → output(1)
    Input features: [T_pyr, T_pyr², √T_pyr]
    Features are standardised before training for stable convergence.

    Parameters
    ----------
    T_pyr        : denoised pyrometer signal (°C)
    T_ref        : thermocouple reference signal (°C)
    cal_fraction : fraction of signal used for training

    Returns
    -------
    T_cal  : np.ndarray — calibrated signal
    coeffs : dict       — trained model and scaler
    """
    cal_end = max(10, int(cal_fraction * len(T_pyr)))
    X_train = _make_features(T_pyr[:cal_end])
    y_train = T_ref[:cal_end].astype(np.float64)

    # Standardise features for better MLP convergence
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_train)

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=1000,
        random_state=42,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
    )
    model.fit(X_sc, y_train)

    X_full = scaler.transform(_make_features(T_pyr))
    T_cal  = model.predict(X_full)

    return T_cal, {
        'method'       : 'MLP',
        'model'        : model,
        'scaler'       : scaler,
        'architecture' : '3→64→32→1',
        'cal_end'      : cal_end,
    }


# =============================================================================
# ML METHOD 7 — GRADIENT BOOSTING
# =============================================================================

def gradient_boosting(T_pyr: np.ndarray,
                      T_ref: np.ndarray,
                      cal_fraction: float = 0.20) -> tuple:
    """
    Gradient Boosting Regression Calibration.

    Builds 100 decision trees sequentially, each correcting the
    residual error of the previous ensemble (boosting). Tends to
    give better accuracy than Random Forest when data is limited.

    Parameters
    ----------
    T_pyr        : denoised pyrometer signal (°C)
    T_ref        : thermocouple reference signal (°C)
    cal_fraction : fraction of signal used for training

    Returns
    -------
    T_cal  : np.ndarray — calibrated signal
    coeffs : dict       — trained model
    """
    cal_end = max(10, int(cal_fraction * len(T_pyr)))
    X_train = _make_features(T_pyr[:cal_end])
    y_train = T_ref[:cal_end].astype(np.float64)

    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    T_cal = model.predict(_make_features(T_pyr))

    return T_cal, {
        'method'       : 'Gradient Boosting',
        'model'        : model,
        'n_estimators' : 100,
        'cal_end'      : cal_end,
    }


# =============================================================================
# ML METHOD 8 — SVR
# =============================================================================

def svr(T_pyr: np.ndarray,
        T_ref: np.ndarray,
        cal_fraction: float = 0.20) -> tuple:
    """
    Support Vector Regression (SVR) Calibration.

    Uses an RBF kernel to find a nonlinear mapping from pyrometer
    reading to true temperature. Effective with small calibration
    datasets — important when limited TC reference points are available.

    Parameters
    ----------
    T_pyr        : denoised pyrometer signal (°C)
    T_ref        : thermocouple reference signal (°C)
    cal_fraction : fraction of signal used for training

    Returns
    -------
    T_cal  : np.ndarray — calibrated signal
    coeffs : dict       — trained model and scaler
    """
    cal_end = max(10, int(cal_fraction * len(T_pyr)))
    X_train = _make_features(T_pyr[:cal_end])
    y_train = T_ref[:cal_end].astype(np.float64)

    # SVR requires standardised features
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X_train)

    model = SVR(kernel='rbf', C=1000, gamma='scale', epsilon=1.0)
    model.fit(X_sc, y_train)

    X_full = scaler.transform(_make_features(T_pyr))
    T_cal  = model.predict(X_full)

    return T_cal, {
        'method'  : 'SVR',
        'model'   : model,
        'scaler'  : scaler,
        'kernel'  : 'RBF',
        'cal_end' : cal_end,
    }


# =============================================================================
# UNIFIED ENTRY POINT
# =============================================================================

# Map method name strings to functions
_METHOD_MAP = {
    'mean_offset'       : mean_offset,
    'linear'            : linear,
    'polynomial'        : polynomial,
    'piecewise_linear'  : piecewise_linear,
    'random_forest'     : random_forest,
    'mlp'               : mlp,
    'gradient_boosting' : gradient_boosting,
    'svr'               : svr,
}


def calibrate(T_pyr: np.ndarray,
              T_ref: np.ndarray,
              method: str = 'linear',
              cal_fraction: float = 0.20) -> tuple:
    """
    Unified calibration entry point — call any of the 8 methods by name.

    Parameters
    ----------
    T_pyr        : denoised pyrometer signal (°C)
    T_ref        : thermocouple reference signal (°C)
    method       : one of METHODS list (default 'linear')
    cal_fraction : fraction of signal used as calibration window

    Returns
    -------
    T_cal  : np.ndarray — calibrated signal
    coeffs : dict       — method-specific parameters

    Example
    -------
    T_cal, c = calibrate(T_pyr, T_tc, method='gradient_boosting')
    """
    if method not in _METHOD_MAP:
        raise ValueError(
            f"Unknown method '{method}'. "
            f"Choose from: {METHODS}"
        )
    return _METHOD_MAP[method](T_pyr, T_ref, cal_fraction=cal_fraction)


def calibrate_all(T_pyr: np.ndarray,
                  T_ref: np.ndarray,
                  cal_fraction: float = 0.20,
                  verbose: bool = True) -> dict:
    """
    Run all 8 calibration methods and return results + metrics.

    Parameters
    ----------
    T_pyr        : denoised pyrometer signal (°C)
    T_ref        : thermocouple reference signal (°C)
    cal_fraction : fraction of signal used as calibration window
    verbose      : if True, print results table

    Returns
    -------
    dict — keys are method names, values are dicts with:
           'T_cal', 'rmse', 'mae', 'max_err', 'time_ms'
    """
    results = {}
    rmse_before = rmse(T_pyr, T_ref)

    if verbose:
        print(f"\n  {'Method':<22} {'RMSE Before':>12} "
              f"{'RMSE After':>11} {'MAE':>8} "
              f"{'MaxErr':>9} {'Time(ms)':>9}")
        print("  " + "-" * 75)

    for name in METHODS:
        t0    = time.perf_counter()
        T_cal, coeffs = calibrate(T_pyr, T_ref, method=name,
                                  cal_fraction=cal_fraction)
        t_ms  = (time.perf_counter() - t0) * 1000

        r_after  = rmse(T_cal, T_ref)
        m_after  = mae(T_cal, T_ref)
        mx_after = max_error(T_cal, T_ref)

        results[name] = {
            'T_cal'   : T_cal,
            'coeffs'  : coeffs,
            'rmse'    : r_after,
            'mae'     : m_after,
            'max_err' : mx_after,
            'time_ms' : t_ms,
        }

        if verbose:
            print(f"  {name:<22} {rmse_before:>12.2f} "
                  f"{r_after:>11.2f} {m_after:>8.2f} "
                  f"{mx_after:>9.2f} {t_ms:>9.1f}")

    if verbose:
        best = min(results, key=lambda k: results[k]['rmse'])
        print(f"\n  ★ Best: {best}  "
              f"(RMSE = {results[best]['rmse']:.2f} °C)")

    return results


# =============================================================================
# SELF-TEST  (run: python calibrate.py)
# =============================================================================

if __name__ == '__main__':
    import os, sys
    import scipy.io as sio
    from scipy.signal import medfilt

    print("=" * 60)
    print("calibrate.py — D2 Self-Test (all 8 ATP-2 methods)")
    print("=" * 60)

    # ── Load NIST Layer01 ─────────────────────────────────────────────
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data (1)', 'data'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'),
        os.path.expanduser('~/Downloads/data (1)/data'),
        os.path.expanduser('~/Downloads/data'),
        os.getcwd(),
        '/mnt/user-data/uploads/'
    ]
    DATA_DIR = sys.argv[1] if len(sys.argv) > 1 else None
    if DATA_DIR is None:
        for c in candidates:
            if os.path.exists(c) and any('Layer01' in f and f.endswith('.mat') for f in os.listdir(c)):
                DATA_DIR = c
                break
    if DATA_DIR is None:
        DATA_DIR = candidates[0]

    mat_file = next(f for f in os.listdir(DATA_DIR)
                    if 'Layer01' in f and f.endswith('.mat'))
    mat  = sio.loadmat(os.path.join(DATA_DIR, mat_file))
    L    = mat['Layer'][0, 0]
    raw3d = L['RadiantTemp'].astype(np.float32)
    sh_A  = float(L['SHvariable_A'].flat[0])
    sh_B  = float(L['SHvariable_B'].flat[0])
    frame_max = raw3d.max(axis=(0, 1))
    T_raw = np.clip(sh_A * frame_max + sh_B - 273.15, 0, 3000)
    T_raw = T_raw[T_raw > 10]
    n     = len(T_raw)

    # ── Simple denoise (median + gaussian) ───────────────────────────
    def _gauss(s, sigma=3):
        w = int(4*sigma+1); x = np.arange(-w, w+1)
        k = np.exp(-0.5*(x/sigma)**2); k /= k.sum()
        return np.convolve(s.astype(np.float64), k, mode='same')

    T_den = _gauss(medfilt(T_raw, 7))

    # ── Simulate thermocouple reference ──────────────────────────────
    np.random.seed(42)
    T_tc = np.zeros(n); T_tc[0] = T_raw[0]
    for i in range(1, n):
        T_tc[i] = T_tc[i-1] + 0.08*(T_raw[i] - T_tc[i-1])
    T_tc += np.random.normal(0, 2, n)

    # ── Run all 8 methods ─────────────────────────────────────────────
    print(f"\n  Signal: {n} frames | "
          f"T=[{T_den.min():.0f}, {T_den.max():.0f}]°C")
    print(f"  Calibration window: first 20% ({int(0.2*n)} frames)\n")
    results = calibrate_all(T_den, T_tc, cal_fraction=0.20, verbose=True)
    print("\n  calibrate.py — all 8 methods working correctly.")
