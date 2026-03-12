# GOAL: get Index(['variable_name', 'input', 'attribute_pool', 'timeseries']) format
# input (i.e.) 'In a Mental Health system, there are 8 metrics:\n Distance is of length 1440: <ts><ts/>;\n Steps is of length 1440: <ts><ts/>;\n Burned Calories is of length 1440: ....
# attribute_pool:
'''
{'seasonal': 
{
'type': 'periodic fluctuation (low-frequency)', 
'amplitude': 0.245, 
'period': 360.0
}, 
'trend': [
{'type': 'increase', 'start_point': 0, 'end_point': 287, 'amplitude': np.float64(5.3916)}, 
{'type': 'decrease', 'start_point': 432, 'end_point': 719, 'amplitude': np.float64(3.3679)}
], 
'frequency': {'type': 'periodic', 'period': 102.4}, 
'noise': {'type': 'high noise', 'std': 0.293}, 
'statistics': {'mean': 2.5463481708333333, 'std': 2.8950460559214086, 'min': 1.0, 'max': 13.550328, 'min_pos': 0, 'max_pos': 374, 'overall_amplitude': 8.71916795, 'overall_bias': 1.0}}

'''

import numpy as np
def get_attribute_data(series):
    x = _sanitize_series(series)
    if x.sum() == 0:
        return {} # skip nan-series
    
    L = x.size

    # 1. statistics
    stats_dict, robust_amp, robust_bias = _basic_statistics(x)
    
    # 2. trend
    trend_dict, trend_line = _trend(x, robust_amp)

    # detrended for frequency and seasonal analysis.
    detrended = x - trend_line

    # 3. frequency
    freq_dict, freq_info = _frequency(detrended, L)

    # 4. seasonal using frequency info
    seasonal_dict, seasonal_comp = _seasonal(detrended, L, freq_info,
                                             base_period=1440, snap_to_base=True,) # some inductive bias for daily cycle
    # 5. noise on residual
    noise_dict, residual = _noise(x, trend_line, seasonal_comp, robust_amp)

    result = {
        "seasonal": seasonal_dict,
        "trend": trend_dict,
        "frequency": freq_dict,
        "noise": noise_dict,
        "statistics": stats_dict
    }

    return result


# ------------- helpers -------------

def _sanitize_series(series, default_value=0.0):
    """
    Convert a time series to a clean 1D float array with no NaN or Inf values.
    Non-finite entries are replaced with the median of finite values.
    If all entries are non-finite, a default value is used.
    """

    x = np.asarray(series, dtype=float).reshape(-1)
    if x.size == 0:
        raise ValueError("Empty time series")

    finite = np.isfinite(x)
    if not np.any(finite):
        med = default_value
    else:
        med = np.nanmedian(x[finite])

    x = np.where(finite, x, med)

    return x    


# def random_area_statistics
def _basic_statistics(x):
    mean_v = float(np.mean(x))
    std_v = float(np.std(x))
    min_v = float(np.min(x))
    max_v = float(np.max(x))
    min_pos = int(np.argmin(x))
    max_pos = int(np.argmax(x))
    q05 = float(np.percentile(x, 5))
    q95 = float(np.percentile(x, 95))
    robust_amp = max(q95 - q05, 1e-8)
    robust_bias = float(np.median(x))
    stats = {
        "mean": round(mean_v, 2),
        "std": round(std_v, 2),
        "min": round(min_v, 2),
        "max": round(max_v, 2),
        "min_pos": min_pos,
        "max_pos": max_pos,
        "overall_amplitude": round(robust_amp, 2),
        "overall_bias": round(robust_bias, 2),
    }
    return stats, robust_amp, robust_bias


import numpy as np

def _trend(x, robust_amp=None, alpha=0.3, tcrit=2.0, win_frac=0.2, step_frac=0.1):
    """
    Detect local increasing/decreasing trends within a signal (scale-invariant).

    Parameters
    ----------
    x : array-like
        Input 1D signal.
    robust_amp : float, optional
        Robust amplitude for normalization; if None, computed from IQR.
    alpha : float
        Minimum normalized trend span required for detection.
    tcrit : float
        Minimum t-statistic of the local slope required for detection.
    win_frac : float
        Fraction of series length used for local window size (0 < win_frac <= 1).
    step_frac : float
        Fractional step size for moving the window.

    Returns
    -------
    out : list of dict
        Each element corresponds to a detected local trend with:
            type: "increase" | "decrease"
            start_point: int (index)
            end_point: int (index)
            amplitude: float (trend span)
    trend_line : np.ndarray
        Global least-squares fit across entire series (for reference).
    """
    x = np.asarray(x, dtype=float)
    L = len(x)
    t = np.arange(L, dtype=float)
    tc = t - t.mean()

    # global fit (for visualization only)
    slope, intercept = np.polyfit(tc, x, deg=1)
    trend_line = slope * tc + intercept

    # global scale normalization
    if robust_amp is None or robust_amp <= 0:
        q25, q75 = np.percentile(x, [25, 75])
        robust_amp = max(q75 - q25, 1e-8)

    # window and step sizes
    win = max(int(L * win_frac), 5)
    step = max(int(L * step_frac), 1)
    if win < 5:
        win = 5

    detected = []

    # slide window and test local slope
    for start in range(0, L - win + 1, step):
        end = start + win
        xi = x[start:end]
        ti = np.arange(win, dtype=float)
        tc = ti - ti.mean()

        # local OLS
        slope, intercept = np.polyfit(tc, xi, deg=1)
        fit_line = slope * tc + intercept
        resid = xi - fit_line

        # robust noise scale
        mad = np.median(np.abs(resid - np.median(resid)))
        robust_sigma = 1.4826 * mad if mad > 1e-12 else np.std(resid)
        robust_sigma = max(robust_sigma, 1e-8)

        # slope t-stat
        Sxx = np.sum(tc**2)
        se_slope = robust_sigma / np.sqrt(Sxx)
        tstat = slope / se_slope if se_slope > 0 else 0.0

        # normalized span strength
        trend_span = abs(slope) * (win - 1)
        strength = trend_span / robust_amp

        if (strength >= alpha) and (abs(tstat) >= tcrit):
            ttype = "increase" if slope > 0 else "decrease"
            detected.append({
                "type": ttype,
                "start_point": int(start),
                "end_point": int(end - 1),
                "amplitude": round(trend_span, 2),
            })

    return detected, trend_line


def _frequency(detrended, L):
    """
    Scale-invariant periodicity detector.
    Assumes input is already detrended.
    """
    y = detrended - detrended.mean()
    nfft = 1 << int(np.ceil(np.log2(max(L, 8))))
    spec = np.fft.rfft(y, n=nfft)
    power = np.abs(spec) ** 2

    if power.size == 0:
        info = {"has_period": False, "freq": 0.0, "peak_ratio": 0.0}
        return {"type": "no periodicity", "period": 0.0}, info

    # remove DC component
    power[0] = 0.0

    # normalize power to remove dependence on signal magnitude
    power /= (power.sum() + 1e-12)

    k = int(np.argmax(power))
    peak_ratio = float(power[k])
    freq = k / nfft
    period = 1.0 / max(freq, 1e-12)

    # validity checks
    valid_period = 2 <= period <= L / 2
    has_period = peak_ratio > 0.05 and valid_period

    info = {
        "has_period": bool(has_period),
        "freq": float(freq),
        "period": float(period),
        "peak_ratio": float(peak_ratio),
    }

    if has_period:
        out = {"type": "periodic", "period": float(round(period, 2))}
    else:
        out = {"type": "no periodicity", "period": 0.0}

    return out, info


def _seasonal(
    detrended,
    L,
    freq_info,
    *,
    base_period=None,          # Optional known base cycle (e.g., 1440 for daily)
    harmonics=(1, 2, 3, 4),    # Relative harmonics of base or detected period
    snap_to_base=False,        # Snap detected freq to base harmonics
    max_harmonic=1,
    suppress_short_period=True,
    min_period_ratio=0.01,     # ignore cycles <1% of total length
    normalize_amplitude=True,
    high_freq_cutoff_ratio=0.25,  # classify high vs low based on period/L ratio
):
    """
    Generalized seasonal component estimation.

    Works with any time scale (minute, hour, week, etc.)
    and supports optional snapping to known base period.

    Parameters:
      base_period: optional known cycle length (e.g. 1440 for daily, 10080 for weekly)
      harmonics: allowed multiples of base if snapping
      snap_to_base: if True, snap to nearest harmonic of base_period
      min_period_ratio: smallest valid period relative to length (e.g., 0.01 = 1%)
      high_freq_cutoff_ratio: defines "high" vs "low" frequency qualitatively
    """
    # --- 1. Early exit for no periodic signal ---
    if not freq_info.get("has_period", False):
        return {
            "type": "no periodic fluctuation",
            "amplitude": 0.0,
        }, np.zeros(L, dtype=float)

    t = np.arange(L, dtype=float)
    freq = freq_info["freq"]
    period = 1.0 / max(freq, 1e-12)

    # --- 2. Snap to known base harmonics (if given) ---
    if snap_to_base and base_period is not None:
        base_freq = 1.0 / base_period
        candidate_freqs = np.array(harmonics) * base_freq
        freq = candidate_freqs[np.argmin(np.abs(candidate_freqs - freq))]
        period = 1.0 / freq

    # --- 3. Optional rejection of too-short cycles ---
    if suppress_short_period and period < L * min_period_ratio:
        return {
            "type": "no periodic fluctuation",
            "amplitude": 0.0,
        }, np.zeros(L, dtype=float)

    # --- 4. Build harmonic basis ---
    cols = []
    for h in range(1, max_harmonic + 1):
        w0 = 2 * np.pi * h * freq
        cols.append(np.cos(w0 * t))
        cols.append(np.sin(w0 * t))
    A = np.column_stack(cols)
    coef, _, _, _ = np.linalg.lstsq(A, detrended, rcond=None)
    seasonal_comp = A @ coef

    # --- 5. Compute amplitude and normalization ---
    a_hat, b_hat = coef[0], coef[1]
    raw_amp = float(np.sqrt(a_hat ** 2 + b_hat ** 2)) * 2.0

    if normalize_amplitude:
        robust_scale = np.percentile(detrended, 75) - np.percentile(detrended, 25)
        if robust_scale <= 1e-12:
            robust_scale = np.std(detrended) + 1e-12
        normalized_amp = raw_amp / robust_scale
    else:
        normalized_amp = raw_amp

    # --- 6. Frequency classification (relative to series length) ---
    high_freq = (period / L) < high_freq_cutoff_ratio
    freq_type = "high" if high_freq else "low"

    out = {
        "type": f"periodic fluctuation ({freq_type}-frequency)",
        "amplitude": float(round(normalized_amp, 3)),
        "period": float(round(period, 2)),
    }
    return out, seasonal_comp


def _noise(x, trend_line, seasonal_comp, robust_amp):
    residual = x - trend_line - seasonal_comp
    noise_std_abs = float(np.std(residual))
    noise_frac = noise_std_abs / max(robust_amp, 1e-8)

    if noise_frac < 0.05:
        ntype = "almost no noise"
    elif noise_frac < 0.1:
        ntype = "low noise"
    elif noise_frac < 0.2:
        ntype = "moderate noise"
    else:
        ntype = "high noise"

    out = {"type": ntype, "std": float(round(noise_frac, 3)),}
    return out, residual
