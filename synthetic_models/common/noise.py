from __future__ import annotations
from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class GaussianNoiseSpec:
    """
    Specification describing additive Gaussian noise.
    """

    name: str
    noise_fraction: float
    seed: int | None = None

    def validate(self) -> None:
        """Validate the noise specification."""

        if not self.name.strip():
            raise ValueError(
                "Noise specification name must not be empty."
            )

        if not np.isfinite(self.noise_fraction):
            raise ValueError(
                "noise_fraction must be finite."
            )

        if self.noise_fraction < 0.0:
            raise ValueError(
                "noise_fraction must be nonnegative."
            )

def calculate_noise_standard_deviation(
    *,
    gravity: np.ndarray,
    noise_fraction: float,
) -> float:
    """
    Calculate the Gaussian noise standard deviation.
    """

    if gravity.ndim != 2:
        raise ValueError(
            "gravity must be a two-dimensional array."
        )

    if not np.all(np.isfinite(gravity)):
        raise ValueError(
            "gravity contains nonfinite values."
        )

    if noise_fraction < 0.0:
        raise ValueError(
            "noise_fraction must be nonnegative."
        )

    signal_peak = float(
        np.max(np.abs(gravity))
    )

    return noise_fraction * signal_peak

def generate_gaussian_noise(
    *,
    gravity: np.ndarray,
    specification: GaussianNoiseSpec,
) -> np.ndarray:
    """
    Generate additive Gaussian noise.
    """

    specification.validate()

    noise_std = calculate_noise_standard_deviation(
        gravity=gravity,
        noise_fraction=specification.noise_fraction,
    )

    generator = np.random.default_rng(
        specification.seed
    )

    noise = generator.normal(
        loc=0.0,
        scale=noise_std,
        size=gravity.shape,
    )

    return np.ascontiguousarray(
        noise,
        dtype=np.float32,
    )

def add_gaussian_noise(
    *,
    gravity: np.ndarray,
    specification: GaussianNoiseSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Add Gaussian noise to a gravity anomaly.
    """

    if gravity.ndim != 2:
        raise ValueError(
            "gravity must be a two-dimensional array."
        )

    if gravity.dtype != np.float32:
        raise TypeError(
            "gravity must have dtype float32."
        )

    if not gravity.flags["C_CONTIGUOUS"]:
        raise ValueError(
            "gravity must be C-contiguous."
        )

    if not np.all(np.isfinite(gravity)):
        raise ValueError(
            "gravity contains nonfinite values."
        )

    noise = generate_gaussian_noise(
        gravity=gravity,
        specification=specification,
    )

    noisy_gravity = gravity + noise

    return (
        np.ascontiguousarray(
            noisy_gravity,
            dtype=np.float32,
        ),
        noise,
    )

def calculate_signal_to_noise_ratio(
    *,
    signal: np.ndarray,
    noise: np.ndarray,
) -> float:
    """
    Calculate root-mean-square signal-to-noise ratio in decibels.
    """

    if signal.shape != noise.shape:
        raise ValueError(
            "signal and noise must have identical shapes."
        )

    if signal.ndim != 2:
        raise ValueError(
            "signal and noise must be two-dimensional arrays."
        )

    if not np.all(np.isfinite(signal)):
        raise ValueError(
            "signal contains nonfinite values."
        )

    if not np.all(np.isfinite(noise)):
        raise ValueError(
            "noise contains nonfinite values."
        )

    signal_float64 = np.asarray(
        signal,
        dtype=np.float64,
    )

    noise_float64 = np.asarray(
        noise,
        dtype=np.float64,
    )

    signal_rms = float(
        np.sqrt(
            np.mean(
                np.square(signal_float64)
            )
        )
    )

    noise_rms = float(
        np.sqrt(
            np.mean(
                np.square(noise_float64)
            )
        )
    )

    if np.isclose(noise_rms, 0.0):
        return float("inf")

    if np.isclose(signal_rms, 0.0):
        return float("-inf")

    return float(
        20.0
        * np.log10(
            signal_rms / noise_rms
        )
    )

def summarize_noise(
    *,
    clean_gravity: np.ndarray,
    noisy_gravity: np.ndarray,
    noise: np.ndarray,
    specification: GaussianNoiseSpec,
) -> dict[str, str | float | int]:
    """
    Calculate summary information for one noisy-gravity realization.
    """

    if clean_gravity.shape != noisy_gravity.shape:
        raise ValueError(
            "clean_gravity and noisy_gravity must have matching shapes."
        )

    if clean_gravity.shape != noise.shape:
        raise ValueError(
            "clean_gravity and noise must have matching shapes."
        )

    for array_name, array in {
        "clean_gravity": clean_gravity,
        "noisy_gravity": noisy_gravity,
        "noise": noise,
    }.items():
        if array.ndim != 2:
            raise ValueError(
                f"{array_name} must be a two-dimensional array."
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(
                f"{array_name} contains nonfinite values."
            )

    signal_rms = float(
        np.sqrt(
            np.mean(
                np.square(
                    clean_gravity.astype(np.float64)
                )
            )
        )
    )

    noise_rms = float(
        np.sqrt(
            np.mean(
                np.square(
                    noise.astype(np.float64)
                )
            )
        )
    )

    requested_noise_std = calculate_noise_standard_deviation(
        gravity=clean_gravity,
        noise_fraction=specification.noise_fraction,
    )

    actual_noise_std = float(
        np.std(
            noise.astype(np.float64)
        )
    )

    correlation = _safe_array_correlation(
        first=clean_gravity,
        second=noisy_gravity,
    )

    snr_db = calculate_signal_to_noise_ratio(
        signal=clean_gravity,
        noise=noise,
    )

    return {
        "noise_name": specification.name,
        "noise_fraction": specification.noise_fraction,
        "noise_percent": 100.0 * specification.noise_fraction,
        "noise_seed": (
            ""
            if specification.seed is None
            else specification.seed
        ),
        "requested_noise_std": requested_noise_std,
        "actual_noise_mean": float(np.mean(noise)),
        "actual_noise_std": actual_noise_std,
        "noise_minimum": float(np.min(noise)),
        "noise_maximum": float(np.max(noise)),
        "signal_rms": signal_rms,
        "noise_rms": noise_rms,
        "snr_db": snr_db,
        "clean_noisy_correlation": correlation,
        "clean_peak_magnitude": float(
            np.max(
                np.abs(clean_gravity)
            )
        ),
        "noisy_peak_magnitude": float(
            np.max(
                np.abs(noisy_gravity)
            )
        ),
    }

def _safe_array_correlation(
    *,
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """
    Return flattened-array correlation.
    """

    first_flat = np.asarray(
        first,
        dtype=np.float64,
    ).reshape(-1)

    second_flat = np.asarray(
        second,
        dtype=np.float64,
    ).reshape(-1)

    if (
        np.isclose(
            np.std(first_flat),
            0.0,
        )
        or np.isclose(
            np.std(second_flat),
            0.0,
        )
    ):
        return float("nan")

    return float(
        np.corrcoef(
            first_flat,
            second_flat,
        )[0, 1]
    )