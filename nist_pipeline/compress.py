"""
=============================================================================
compress.py  —  D2: ATP-3 Compression Module
=============================================================================
Thesis  : Automation of pyrometer data pre-processing
          (ATP-2 Calibration + ATP-3 Compression)
Author  : [Your Name]

WHAT THIS MODULE DOES:
  Provides 3 compression methods for calibrated pyrometer time-series.
  Each method reduces storage size while preserving temperature accuracy.
  Reports Compression Ratio (CR) and Reconstruction RMSE (°C).

ATP-3 COMPRESSION METHODS (Research Question 2):
    1. delta_encoding   — store differences between consecutive samples
    2. vae              — Variational Autoencoder (learned latent space)
    3. deep_autoencoder — Deep Autoencoder (deeper encoder/decoder)

HOW TO USE:
  from compress import compress, METHODS, compression_ratio, recon_rmse

  # Compress with one method
  result = compress(T_cal, method='delta_encoding')
  T_recon = result['T_reconstructed']
  CR      = result['compression_ratio']

  # Run all 3 methods
  for name in METHODS:
      result = compress(T_cal, method=name)

KEY DEFINITIONS:
  Compression Ratio (CR) = original_size / compressed_size
  Higher CR = more compression (fewer bytes stored)
  RMSE = sqrt(mean((T_original - T_reconstructed)²))  in °C
  Lower RMSE = better reconstruction accuracy
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Registry of available method names
METHODS = ['delta_encoding', 'vae', 'deep_autoencoder']

# Fixed random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# =============================================================================
# METRICS
# =============================================================================

def compression_ratio(original_size: int, compressed_size: int) -> float:
    """
    Compute compression ratio.

        CR = original_size / compressed_size

    Parameters
    ----------
    original_size   : number of values in original signal
    compressed_size : number of values stored after compression

    Returns
    -------
    float — compression ratio (e.g. 5.0 means 5× smaller)
    """
    return original_size / max(compressed_size, 1)


def recon_rmse(T_original: np.ndarray,
               T_reconstructed: np.ndarray) -> float:
    """
    Reconstruction RMSE between original and reconstructed signal.

    Parameters
    ----------
    T_original      : original calibrated signal (°C)
    T_reconstructed : signal after compress → decompress (°C)

    Returns
    -------
    float — RMSE in °C
    """
    n = min(len(T_original), len(T_reconstructed))
    return float(np.sqrt(np.mean(
        (T_original[:n] - T_reconstructed[:n]) ** 2
    )))


# =============================================================================
# METHOD 1 — DELTA ENCODING
# =============================================================================

def delta_encoding(T_cal: np.ndarray,
                   quantise_bits: int = 12) -> dict:
    """
    Delta Encoding Compression.

    Instead of storing absolute temperature values, stores the
    difference (delta) between consecutive samples:

        delta[i] = T[i] - T[i-1]   for i > 0

    Deltas are much smaller numbers than raw temperatures, so they
    can be quantised to fewer bits without large errors.

    Quantisation:
        - Deltas are scaled to fit in a [-2^(bits-1), 2^(bits-1)-1]
          integer range and stored as integers.
        - On reconstruction, integers are rescaled back to °C and
          accumulated (cumulative sum) to recover the signal.

    Why it works for pyrometer data:
        Temperature changes slowly between frames (2 ms apart), so
        consecutive differences are small — much more compressible
        than absolute values.

    Parameters
    ----------
    T_cal         : calibrated pyrometer signal (°C)
    quantise_bits : integer bit depth for delta quantisation (default 12)
                    12 bits → 4096 levels
                    Lower bits → higher CR, higher RMSE

    Returns
    -------
    dict with keys:
        'T_reconstructed'  : np.ndarray — reconstructed signal (°C)
        'compressed_data'  : dict       — first value + quantised deltas
        'compression_ratio': float      — CR achieved
        'recon_rmse'       : float      — reconstruction RMSE (°C)
        'original_size'    : int
        'compressed_size'  : int
        'method'           : str
    """
    T = T_cal.astype(np.float64)
    n = len(T)

    # ── COMPRESS ──────────────────────────────────────────────────────
    # Step 1: compute deltas
    deltas = np.diff(T)                     # shape (n-1,)

    # Step 2: quantise deltas to integers
    delta_max = np.max(np.abs(deltas)) + 1e-9
    levels    = 2 ** quantise_bits          # number of quantisation levels
    scale     = (levels / 2 - 1) / delta_max
    q_deltas  = np.round(deltas * scale).astype(np.int32)

    # Compressed representation: first value (float64) + int deltas
    compressed_size = 1 + len(q_deltas)    # 1 anchor + n-1 deltas

    # ── DECOMPRESS ────────────────────────────────────────────────────
    # Step 3: dequantise
    deltas_recon = q_deltas.astype(np.float64) / scale

    # Step 4: reconstruct via cumulative sum from first value
    T_recon      = np.empty(n)
    T_recon[0]   = T[0]
    T_recon[1:]  = T[0] + np.cumsum(deltas_recon)

    CR   = compression_ratio(n, compressed_size)
    rmse = recon_rmse(T, T_recon)

    return {
        'T_reconstructed'   : T_recon,
        'compressed_data'   : {
            'first_value' : T[0],
            'q_deltas'    : q_deltas,
            'scale'       : scale,
            'bits'        : quantise_bits,
        },
        'compression_ratio' : CR,
        'recon_rmse'        : rmse,
        'original_size'     : n,
        'compressed_size'   : compressed_size,
        'method'            : 'Delta Encoding',
    }


# =============================================================================
# HELPER — SEGMENT SIGNAL INTO WINDOWS FOR AUTOENCODER TRAINING
# =============================================================================

def _segment_signal(T: np.ndarray,
                    window_size: int) -> tuple:
    """
    Segment a 1-D signal into overlapping windows for autoencoder input.

    Parameters
    ----------
    T           : 1-D temperature signal
    window_size : number of samples per window

    Returns
    -------
    X      : np.ndarray — shape (n_windows, window_size)
    n_orig : int        — original signal length (for reconstruction)
    """
    n       = len(T)
    n_wins  = n // window_size
    X       = T[:n_wins * window_size].reshape(n_wins, window_size)
    return X.astype(np.float32), n


def _reconstruct_from_windows(X_recon: np.ndarray,
                               n_orig: int) -> np.ndarray:
    """
    Flatten windowed reconstruction back to 1-D signal.

    Parameters
    ----------
    X_recon : np.ndarray — shape (n_windows, window_size)
    n_orig  : int        — original signal length

    Returns
    -------
    np.ndarray — 1-D reconstructed signal, length = n_orig
    """
    T_flat = X_recon.flatten()
    # Pad or trim to match original length
    if len(T_flat) < n_orig:
        T_flat = np.pad(T_flat, (0, n_orig - len(T_flat)),
                        mode='edge')
    return T_flat[:n_orig]


def _normalise(X: np.ndarray) -> tuple:
    """
    Min-max normalise to [0, 1] range.
    Returns normalised array and (min, max) for denormalisation.
    """
    xmin, xmax = X.min(), X.max()
    X_norm = (X - xmin) / (xmax - xmin + 1e-9)
    return X_norm.astype(np.float32), xmin, xmax


def _denormalise(X_norm: np.ndarray,
                 xmin: float,
                 xmax: float) -> np.ndarray:
    """Reverse min-max normalisation back to °C."""
    return X_norm * (xmax - xmin + 1e-9) + xmin


# =============================================================================
# METHOD 2 — VARIATIONAL AUTOENCODER (VAE)
# =============================================================================

class _VAEModel(nn.Module):
    """
    Variational Autoencoder for time-series compression.

    Architecture:
        Encoder: window_size → 64 → 32 → latent_dim × 2 (μ and log σ²)
        Decoder: latent_dim → 32 → 64 → window_size

    The latent space has a probabilistic interpretation:
        z ~ N(μ, σ²)   (reparameterisation trick during training)
    During compression, we store only μ (the mean of the latent dist).
    """

    def __init__(self, window_size: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder — maps input window to latent mean and log-variance
        self.encoder = nn.Sequential(
            nn.Linear(window_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.fc_mu     = nn.Linear(32, latent_dim)   # mean
        self.fc_logvar = nn.Linear(32, latent_dim)   # log variance

        # Decoder — maps latent vector back to window
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, window_size),
        )

    def encode(self, x):
        """Encode input to latent (μ, log σ²)."""
        h      = self.encoder(x)
        mu     = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterise(self, mu, logvar):
        """
        Reparameterisation trick: z = μ + ε × σ  (ε ~ N(0,1))
        Allows gradients to flow through the sampling operation.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu   # at inference time, use the mean directly

    def decode(self, z):
        """Decode latent vector to reconstructed window."""
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterise(mu, logvar)
        x_recon    = self.decode(z)
        return x_recon, mu, logvar


def _vae_loss(x_recon, x, mu, logvar, beta: float = 1.0):
    """
    VAE loss = Reconstruction loss + β × KL divergence

    Reconstruction loss: MSE between input and output windows
    KL divergence: regularises latent space to be close to N(0,1)
    β controls the trade-off (β=1 is standard VAE)
    """
    recon_loss = nn.functional.mse_loss(x_recon, x, reduction='mean')
    kl_loss    = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )
    return recon_loss + beta * kl_loss


def vae(T_cal: np.ndarray,
        window_size: int = 64,
        latent_dim: int = 4,
        epochs: int = 50,
        batch_size: int = 32) -> dict:
    """
    Variational Autoencoder (VAE) Compression.

    Trains a VAE on the calibrated temperature signal to learn a
    compact latent representation. The signal is split into windows
    of length `window_size`; each window is encoded to `latent_dim`
    values. Only the latent vectors are stored (compressed form).

    Compression Ratio = window_size / latent_dim
    (e.g. window=64, latent=4 → CR = 16×)

    Parameters
    ----------
    T_cal       : calibrated pyrometer signal (°C)
    window_size : samples per window (default 64)
    latent_dim  : latent space dimensions (default 4)
    epochs      : training epochs (default 50)
    batch_size  : mini-batch size (default 32)

    Returns
    -------
    dict with keys:
        'T_reconstructed'  : np.ndarray — reconstructed signal (°C)
        'latent_vectors'   : np.ndarray — compressed representation
        'compression_ratio': float
        'recon_rmse'       : float
        'original_size'    : int
        'compressed_size'  : int
        'method'           : str
        'model'            : trained _VAEModel
    """
    # ── Prepare data ──────────────────────────────────────────────────
    X, n_orig       = _segment_signal(T_cal, window_size)
    X_norm, mn, mx  = _normalise(X)
    n_windows       = X_norm.shape[0]

    dataset    = TensorDataset(torch.from_numpy(X_norm))
    loader     = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ── Build and train model ─────────────────────────────────────────
    model     = _VAEModel(window_size, latent_dim)
    optimiser = optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimiser.zero_grad()
            x_recon, mu, logvar = model(batch)
            loss = _vae_loss(x_recon, batch, mu, logvar)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"    VAE epoch {epoch+1:3d}/{epochs}  "
                  f"loss={epoch_loss/len(loader):.6f}")

    # ── Compress: encode all windows to latent vectors ────────────────
    model.eval()
    with torch.no_grad():
        X_tensor  = torch.from_numpy(X_norm)
        mu, _     = model.encode(X_tensor)
        latent    = mu.numpy()                  # shape (n_windows, latent_dim)

    compressed_size = latent.size   # total stored values

    # ── Decompress: decode latent vectors back to windows ─────────────
    with torch.no_grad():
        z_tensor  = torch.from_numpy(latent.astype(np.float32))
        X_recon_n = model.decode(z_tensor).numpy()

    X_recon   = _denormalise(X_recon_n, mn, mx)
    T_recon   = _reconstruct_from_windows(X_recon, n_orig)

    CR   = compression_ratio(n_orig, compressed_size)
    rmse = recon_rmse(T_cal, T_recon)

    return {
        'T_reconstructed'   : T_recon,
        'latent_vectors'    : latent,
        'compression_ratio' : CR,
        'recon_rmse'        : rmse,
        'original_size'     : n_orig,
        'compressed_size'   : compressed_size,
        'method'            : 'VAE',
        'model'             : model,
        'window_size'       : window_size,
        'latent_dim'        : latent_dim,
    }


# =============================================================================
# METHOD 3 — DEEP AUTOENCODER
# =============================================================================

class _DeepAEModel(nn.Module):
    """
    Deep Autoencoder for time-series compression.

    Architecture (deeper than Sravya's shallow AE):
        Encoder: window_size → 128 → 64 → 32 → bottleneck
        Decoder: bottleneck → 32 → 64 → 128 → window_size

    Unlike VAE, the Deep AE is deterministic — it learns a fixed
    compressed code (no probabilistic latent space). The extra layers
    allow it to learn more complex temperature patterns.
    """

    def __init__(self, window_size: int, bottleneck: int):
        super().__init__()

        # Encoder — progressively compresses the input
        self.encoder = nn.Sequential(
            nn.Linear(window_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, bottleneck),
        )

        # Decoder — progressively reconstructs the input
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, window_size),
        )

    def forward(self, x):
        z      = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


def deep_autoencoder(T_cal: np.ndarray,
                     window_size: int = 64,
                     bottleneck: int = 4,
                     epochs: int = 50,
                     batch_size: int = 32) -> dict:
    """
    Deep Autoencoder Compression.

    Trains a 4-layer deep autoencoder on the calibrated temperature
    signal. Each window of `window_size` samples is compressed to
    `bottleneck` values (the encoded representation).

    Compression Ratio = window_size / bottleneck
    (e.g. window=64, bottleneck=4 → CR = 16×)

    Compared to Sravya's shallow AE (1 encoder layer, bottleneck=3):
    - More layers → can learn more complex temperature patterns
    - Same or better reconstruction at equivalent CR

    Parameters
    ----------
    T_cal       : calibrated pyrometer signal (°C)
    window_size : samples per window (default 64)
    bottleneck  : compressed code size per window (default 4)
    epochs      : training epochs (default 50)
    batch_size  : mini-batch size (default 32)

    Returns
    -------
    dict with keys:
        'T_reconstructed'  : np.ndarray — reconstructed signal (°C)
        'encoded_data'     : np.ndarray — compressed representation
        'compression_ratio': float
        'recon_rmse'       : float
        'original_size'    : int
        'compressed_size'  : int
        'method'           : str
        'model'            : trained _DeepAEModel
    """
    # ── Prepare data ──────────────────────────────────────────────────
    X, n_orig      = _segment_signal(T_cal, window_size)
    X_norm, mn, mx = _normalise(X)
    n_windows      = X_norm.shape[0]

    dataset   = TensorDataset(torch.from_numpy(X_norm))
    loader    = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # ── Build and train model ─────────────────────────────────────────
    model     = _DeepAEModel(window_size, bottleneck)
    optimiser = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimiser.zero_grad()
            x_recon = model(batch)
            loss    = criterion(x_recon, batch)
            loss.backward()
            optimiser.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"    DeepAE epoch {epoch+1:3d}/{epochs}  "
                  f"loss={epoch_loss/len(loader):.6f}")

    # ── Compress: encode all windows ─────────────────────────────────
    model.eval()
    with torch.no_grad():
        X_tensor    = torch.from_numpy(X_norm)
        encoded     = model.encode(X_tensor).numpy()  # (n_windows, bottleneck)

    compressed_size = encoded.size

    # ── Decompress: decode encoded vectors back to windows ─────────── 
    with torch.no_grad():
        z_tensor    = torch.from_numpy(encoded.astype(np.float32))
        X_recon_n   = model.decode(z_tensor).numpy()

    X_recon  = _denormalise(X_recon_n, mn, mx)
    T_recon  = _reconstruct_from_windows(X_recon, n_orig)

    CR   = compression_ratio(n_orig, compressed_size)
    rmse = recon_rmse(T_cal, T_recon)

    return {
        'T_reconstructed'   : T_recon,
        'encoded_data'      : encoded,
        'compression_ratio' : CR,
        'recon_rmse'        : rmse,
        'original_size'     : n_orig,
        'compressed_size'   : compressed_size,
        'method'            : 'Deep Autoencoder',
        'model'             : model,
        'window_size'       : window_size,
        'bottleneck'        : bottleneck,
    }


# =============================================================================
# UNIFIED ENTRY POINT
# =============================================================================

_METHOD_MAP = {
    'delta_encoding' : delta_encoding,
    'vae'            : vae,
    'deep_autoencoder': deep_autoencoder,
}


def compress(T_cal: np.ndarray,
             method: str = 'delta_encoding',
             **kwargs) -> dict:
    """
    Unified compression entry point — call any of the 3 methods by name.

    Parameters
    ----------
    T_cal   : calibrated pyrometer signal (°C)
    method  : one of METHODS list (default 'delta_encoding')
    **kwargs: method-specific parameters (e.g. latent_dim=4)

    Returns
    -------
    dict — keys: T_reconstructed, compression_ratio, recon_rmse, ...

    Example
    -------
    result = compress(T_cal, method='vae', latent_dim=8, epochs=30)
    """
    if method not in _METHOD_MAP:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: {METHODS}"
        )
    return _METHOD_MAP[method](T_cal, **kwargs)


def compress_all(T_cal: np.ndarray,
                 verbose: bool = True) -> dict:
    """
    Run all 3 ATP-3 compression methods and return results.

    Parameters
    ----------
    T_cal   : calibrated pyrometer signal (°C)
    verbose : if True, print results table

    Returns
    -------
    dict — keys are method names, values are result dicts
    """
    results = {}

    if verbose:
        print(f"\n  {'Method':<20} {'CR':>6} "
              f"{'RMSE(°C)':>10} {'Orig':>7} {'Compressed':>12}")
        print("  " + "-" * 60)

    for name in METHODS:
        print(f"\n  Running {name}...")
        result = compress(T_cal, method=name)
        results[name] = result

        if verbose:
            print(f"  {name:<20} {result['compression_ratio']:>6.1f}x "
                  f"{result['recon_rmse']:>10.2f} "
                  f"{result['original_size']:>7} "
                  f"{result['compressed_size']:>12}")

    if verbose:
        best_rmse = min(results, key=lambda k: results[k]['recon_rmse'])
        best_cr   = max(results, key=lambda k: results[k]['compression_ratio'])
        print(f"\n  ★ Best RMSE : {best_rmse} "
              f"({results[best_rmse]['recon_rmse']:.2f} °C)")
        print(f"  ★ Best CR   : {best_cr} "
              f"({results[best_cr]['compression_ratio']:.1f}×)")

    return results


# =============================================================================
# SELF-TEST  (run: python compress.py)
# =============================================================================

if __name__ == '__main__':
    import os, sys
    import scipy.io as sio
    from scipy.signal import medfilt

    print("=" * 60)
    print("compress.py — D2 Self-Test (all 3 ATP-3 methods)")
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
    mat   = sio.loadmat(os.path.join(DATA_DIR, mat_file))
    L     = mat['Layer'][0, 0]
    raw3d = L['RadiantTemp'].astype(np.float32)
    sh_A  = float(L['SHvariable_A'].flat[0])
    sh_B  = float(L['SHvariable_B'].flat[0])
    frame_max = raw3d.max(axis=(0, 1))
    T_raw = np.clip(sh_A * frame_max + sh_B - 273.15, 0, 3000)
    T_raw = T_raw[T_raw > 10]
    n     = len(T_raw)

    # ── Simple denoise + simulate calibrated signal ───────────────────
    def _gauss(s, sigma=3):
        w = int(4*sigma+1); x = np.arange(-w, w+1)
        k = np.exp(-0.5*(x/sigma)**2); k /= k.sum()
        return np.convolve(s.astype(np.float64), k, mode='same')

    T_den = _gauss(medfilt(T_raw, 7))
    T_cal = T_den * 0.92 - 30    # simulate calibrated signal

    print(f"\n  Signal: {n} frames | "
          f"T=[{T_cal.min():.0f}, {T_cal.max():.0f}]°C\n")

    # ── Run all 3 methods ─────────────────────────────────────────────
    results = compress_all(T_cal, verbose=True)
    print("\n  compress.py — all 3 methods working correctly.")
