"""Carga y utilidades del modelo GAF-Net para detección de enfermedades en maíz.

Este módulo adapta el notebook original a una aplicación Streamlit. La función
principal es `load_gafnet_model()`, que registra los módulos personalizados
GSConv y FASFF, prepara el checkpoint y devuelve un objeto Ultralytics YOLO.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import zipfile
from copy import deepcopy
from typing import Iterable

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO
from ultralytics.nn import tasks

DEFAULT_WEIGHTS_URL = (
    "https://github.com/AnanyaGubba/"
    "GAF-Net-Multi-Disease-Corn-Leaf-Detection/"
    "raw/refs/heads/main/best.pt.zip"
)

LOCAL_MODEL_PATH = pathlib.Path("models/best.pt")
DEFAULT_CACHE_DIR = pathlib.Path(os.getenv("GAFNET_CACHE_DIR", "model_cache"))


class GSConv(nn.Module):
    """Grouped Spatial Convolution usado por el checkpoint de GAF-Net."""

    def __init__(self, c1: int, c2: int | None = None, k: int = 3, s: int = 1):
        super().__init__()
        if c2 is None:
            c2 = c1

        c_ = c2 // 2
        self.cv1 = nn.Conv2d(c1, c_, 1, s)
        self.dw = nn.Conv2d(c_, c_, k, 1, padding=k // 2, groups=c_)
        self.bn = nn.BatchNorm2d(c_)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.cv1(x)
        x2 = self.act(self.bn(self.dw(x1)))
        y = torch.cat((x1, x2), dim=1)
        b, c, h, w = y.shape
        return y.reshape(b, 2, c // 2, h, w).permute(0, 2, 1, 3, 4).reshape(b, c, h, w)


class FASFF(nn.Module):
    """Feature-Adaptive Spatial Feature Fusion usado por el checkpoint."""

    def __init__(self, c1: int | list[int], c2: int = 256, level: int = 0):
        super().__init__()
        if isinstance(c1, int):
            c1 = [c1, c1, c1]

        self.level = level
        self.c2 = c2
        self.reduce = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ci, c2, 1, 1, bias=False),
                    nn.BatchNorm2d(c2),
                    nn.SiLU(),
                )
                for ci in c1
            ]
        )
        self.weight_layers = nn.ModuleList([nn.Conv2d(c2, 1, 1, 1, 0) for _ in range(3)])

    @staticmethod
    def _resize_to(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] == size:
            return x
        if x.shape[-2] > size[0] or x.shape[-1] > size[1]:
            return F.adaptive_avg_pool2d(x, size)
        return F.interpolate(x, size=size, mode="nearest")

    def forward(self, x: Iterable[torch.Tensor]) -> torch.Tensor:
        x = list(x)
        if len(x) != 3:
            raise ValueError("FASFF espera exactamente tres mapas de características.")

        target_size = x[self.level].shape[-2:]
        feats = [self.reduce[i](self._resize_to(x[i], target_size)) for i in range(3)]
        weights = torch.stack([self.weight_layers[i](feats[i]) for i in range(3)], dim=0)
        weights = torch.softmax(weights, dim=0)
        return weights[0] * feats[0] + weights[1] * feats[1] + weights[2] * feats[2]


def _get_out_channels(module: nn.Module) -> int | None:
    if isinstance(module, GSConv):
        return module.cv1.out_channels * 2
    if isinstance(module, nn.Sequential) and len(module) > 0:
        return _get_out_channels(module[-1])
    if hasattr(module, "cv2"):
        return _get_out_channels(module.cv2)
    if hasattr(module, "conv") and isinstance(module.conv, nn.Conv2d):
        return module.conv.out_channels
    for child in reversed(list(module.children())):
        result = _get_out_channels(child)
        if result is not None:
            return result
    return None


def register_gafnet_modules() -> None:
    """Registra GSConv y FASFF para que Ultralytics pueda reconstruir el modelo."""
    tasks.GSConv = GSConv
    tasks.FASFF = FASFF

    # Algunos checkpoints guardados desde notebooks buscan las clases en __main__.
    import __main__

    __main__.GSConv = GSConv
    __main__.FASFF = FASFF

    try:
        torch.serialization.add_safe_globals(
            [
                GSConv,
                FASFF,
                nn.Conv2d,
                nn.BatchNorm2d,
                nn.SiLU,
                nn.ModuleList,
                nn.Sequential,
            ]
        )
    except Exception:
        pass

    if getattr(tasks.parse_model, "__name__", "") == "patched_parse_model":
        return

    if not hasattr(tasks, "_original_parse_model"):
        tasks._original_parse_model = tasks.parse_model

    def patched_parse_model(d, ch, verbose=True):
        d2 = deepcopy(d)
        fasff_info = []
        gsconv_info = []

        for section in ("backbone", "head"):
            for i, layer in enumerate(d2.get(section, [])):
                f, n, m_name, args = layer
                if m_name == "FASFF":
                    fasff_info.append((section, i, f, args, n))
                    d2[section][i] = [-1, 1, "Conv", [args[0], 1, 1]]
                elif m_name == "GSConv":
                    gsconv_info.append((section, i, f, args, n))
                    d2[section][i] = [f, n, "Conv", args]

        model, save = tasks._original_parse_model(d2, ch, verbose)

        if not gsconv_info and not fasff_info:
            return model, save

        backbone_len = len(d.get("backbone", []))

        for section, idx, from_val, orig_args, _n in gsconv_info:
            global_idx = idx if section == "backbone" else backbone_len + idx
            dummy = model[global_idx]
            conv_mod = dummy[0] if isinstance(dummy, nn.Sequential) else dummy

            c1_actual = conv_mod.conv.in_channels
            c2_actual = conv_mod.conv.out_channels
            k = orig_args[1] if len(orig_args) > 1 else 3
            s = orig_args[2] if len(orig_args) > 2 else 1
            n_actual = len(dummy) if isinstance(dummy, nn.Sequential) else 1

            if n_actual > 1:
                gs = nn.Sequential(
                    *(GSConv(c1_actual if j == 0 else c2_actual, c2_actual, k, s if j == 0 else 1) for j in range(n_actual))
                )
            else:
                gs = GSConv(c1_actual, c2_actual, k, s)

            gs.i = global_idx
            gs.f = from_val
            gs.type = "GSConv"
            gs.np = sum(p.numel() for p in gs.parameters())
            model[global_idx] = gs

        for section, idx, from_val, args, _n in fasff_info:
            global_idx = idx if section == "backbone" else backbone_len + idx
            if isinstance(from_val, list):
                input_chs = [_get_out_channels(model[fi]) for fi in from_val]
            else:
                input_chs = [_get_out_channels(model[from_val])] * 3

            actual_c2 = _get_out_channels(model[global_idx])
            level = args[1] if len(args) > 1 else 0
            fasff = FASFF(input_chs, actual_c2, level)

            fasff.i = global_idx
            fasff.f = from_val
            fasff.type = "FASFF"
            fasff.np = sum(p.numel() for p in fasff.parameters())
            model[global_idx] = fasff

            if isinstance(from_val, list):
                for fidx in from_val:
                    real = fidx if fidx >= 0 else global_idx + fidx
                    if real not in save:
                        save.append(real)

        return model, save

    tasks.parse_model = patched_parse_model


def looks_like_torch_checkpoint_zip(path: pathlib.Path) -> bool:
    """Detecta si un ZIP parece ser directamente un checkpoint torch.save()."""
    if not zipfile.is_zipfile(path):
        return False
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = zf.namelist()
        has_data_pkl = any(name.endswith("/data.pkl") or name == "data.pkl" for name in names)
        has_tensor_data = any("/data/" in name for name in names)
        return has_data_pkl and has_tensor_data
    except Exception:
        return False


def download_file(url: str, destination: pathlib.Path, timeout: int = 180) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with destination.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def prepare_checkpoint(cache_dir: pathlib.Path = DEFAULT_CACHE_DIR, weights_url: str | None = None) -> pathlib.Path:
    """Devuelve la ruta local a `best.pt`.

    Prioridad:
    1. `models/best.pt`, si existe.
    2. `model_cache/best.pt`, si ya fue preparado.
    3. Descarga remota de `best.pt.zip` y conversión robusta a `best.pt`.
    """
    if LOCAL_MODEL_PATH.exists():
        return LOCAL_MODEL_PATH

    weights_url = weights_url or os.getenv("GAFNET_WEIGHTS_URL", DEFAULT_WEIGHTS_URL)
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    zip_path = cache_dir / "best.pt.zip"
    best_pt = cache_dir / "best.pt"

    if best_pt.exists() and best_pt.stat().st_size > 0:
        return best_pt

    if not zip_path.exists() or zip_path.stat().st_size == 0:
        download_file(weights_url, zip_path)

    if looks_like_torch_checkpoint_zip(zip_path):
        shutil.copy2(zip_path, best_pt)
    else:
        extract_dir = cache_dir / "weights"
        extract_dir.mkdir(parents=True, exist_ok=True)
        if not zipfile.is_zipfile(zip_path):
            raise RuntimeError("El archivo descargado no es un ZIP válido ni un checkpoint PyTorch reconocible.")

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

        pt_files = list(extract_dir.rglob("*.pt"))
        if pt_files:
            shutil.copy2(pt_files[0], best_pt)
        else:
            candidates = [p for p in extract_dir.rglob("*") if p.is_file()]
            torch_candidates = [p for p in candidates if looks_like_torch_checkpoint_zip(p)]
            if torch_candidates:
                shutil.copy2(max(torch_candidates, key=lambda p: p.stat().st_size), best_pt)
            else:
                raise FileNotFoundError("No se pudo identificar automáticamente el checkpoint del modelo.")

    return best_pt


def load_gafnet_model():
    """Registra los módulos personalizados, prepara el checkpoint y carga YOLO."""
    register_gafnet_modules()
    checkpoint = prepare_checkpoint()
    try:
        return YOLO(str(checkpoint))
    except Exception:
        # Segundo intento para entornos PyTorch estrictos con safe globals.
        register_gafnet_modules()
        return YOLO(str(checkpoint))
