import contextlib
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Generator, BinaryIO
import copy
import io
import os
import tempfile
import warnings
from contextlib import contextmanager
from functools import partial
from hashlib import sha256
from pathlib import Path
import requests
import json
import enum
from ..future import Future

class ModelType(enum.IntEnum):
    """
    Inferred model type from the U-Net `in_channels`.
    """

    UNKNOWN = 0
    PROMPT_TO_IMAGE = 4
    DEPTH = 5
    UPSCALING = 7
    INPAINTING = 9

    CONTROL_NET = -1

    @classmethod
    def _missing_(cls, _):
        return cls.UNKNOWN

    def recommended_model(self) -> str:
        """Provides a recommended model for a given task.

        This method has a bias towards the latest version of official Stability AI models.
        """
        match self:
            case ModelType.PROMPT_TO_IMAGE:
                return "stabilityai/stable-diffusion-2-1"
            case ModelType.DEPTH:
                return "stabilityai/stable-diffusion-2-depth"
            case ModelType.UPSCALING:
                return "stabilityai/stable-diffusion-x4-upscaler"
            case ModelType.INPAINTING:
                return "stabilityai/stable-diffusion-2-inpainting"
            case _:
                return "stabilityai/stable-diffusion-2-1"


@dataclass
class Model:
    id: str
    author: str
    tags: list[str]
    likes: int
    downloads: int
    model_type: ModelType


def hf_list_models(
    self,
    query: str,
    token: str,
) -> list[Model]:
    from huggingface_hub import HfApi, ModelFilter

    if hasattr(self, "huggingface_hub_api"):
        api: HfApi = self.huggingface_hub_api
    else:
        api = HfApi()
        setattr(self, "huggingface_hub_api", api)

    filter = ModelFilter(tags="diffusers", task="text-to-image")
    models = api.list_models(filter=filter, search=query, use_auth_token=token)
    return [
        Model(
            m.modelId,
            m.author or "",
            m.tags,
            m.likes if hasattr(m, "likes") else 0,
            getattr(m, "downloads", -1),
            ModelType.UNKNOWN,
        )
        for m in models
        if m.modelId is not None
        and m.tags is not None
        and "diffusers" in (m.tags or {})
    ]


def hf_list_installed_models(self) -> list[Model]:
    from diffusers.utils import DIFFUSERS_CACHE

    DIFFUSERS_CACHE = Path(DIFFUSERS_CACHE)


    def list_dir(cache_dir):
        if not cache_dir.exists():
            return []

    def detect_model_type(snapshot_folder):
        unet_config = os.path.join(snapshot_folder, 'unet', 'config.json')
        config = os.path.join(snapshot_folder, 'config.json')
        if os.path.exists(unet_config):
            with open(unet_config, 'r') as f:
                return ModelType(json.load(f)['in_channels'])
        elif os.path.exists(config):
            with open(config, 'r') as f:
                config_dict = json.load(f)
                if '_class_name' in config_dict and config_dict['_class_name'] == 'ControlNetModel':
                    return ModelType.CONTROL_NET
                else:
                    return ModelType.UNKNOWN
        else:
            return ModelType.UNKNOWN

    def _map_model(file):
        storage_folder = DIFFUSERS_CACHE / file
        snapshot_folder = None
        model_type = ModelType.UNKNOWN

        if (storage_folder / "model_index.json").exists():
            snapshot_folder = storage_folder
            model_type = detect_model_type(snapshot_folder)
        else:

            ref_path = storage_folder / "refs"
            for revision in ref_path.iterdir():
                ref_path = storage_folder / "refs" / revision
                if ref_path.exists():
                    commit_hash = ref_path.read_text()
                    snapshot_folder = storage_folder / "snapshots" / commit_hash
                    if (
                        detected_type := detect_model_type(snapshot_folder)
                    ) != ModelType.UNKNOWN:
                        model_type = detected_type
                        break
            # else:
            #     print(f"Could not find {revision} in {ref_path.parent.as_posix()}")
            #     print(f"Candidates: {[x.stem for x in ref_path.parent.iterdir()]}")
        model_type = ModelType.UNKNOWN

        if snapshot_folder:
            with contextlib.suppress(Exception):
                with open(snapshot_folder / "unet" / "config.json", "r") as f:
                    model_type = ModelType(json.load(f)["in_channels"])

        return Model(storage_folder.as_posix(), "", [], -1, -1, model_type)

    return [_map_model(file) for file in DIFFUSERS_CACHE.iterdir() if file.is_dir()]


@dataclass
class DownloadStatus:
    file: str
    index: int
    total: int


def hf_snapshot_download(
    self,
    model: str,
    token: str,
    revision: str | None = None
):
    from huggingface_hub import utils

    future = Future()
    yield future

    class future_tqdm(utils.tqdm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            future.add_response(DownloadStatus(self.desc, 0, self.total))

        def update(self, n=1):
            future.add_response(DownloadStatus(self.desc, self.last_print_n + n, self.total))
            return super().update(n=n)
    
    from huggingface_hub import file_download
    file_download.tqdm = future_tqdm
    from huggingface_hub import _snapshot_download
    
    from diffusers import StableDiffusionPipeline
    from diffusers.utils import (
        DIFFUSERS_CACHE,
        WEIGHTS_NAME,
        CONFIG_NAME,
        ONNX_WEIGHTS_NAME,
    )
    from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
    
    try:
        config_dict = StableDiffusionPipeline.load_config(
            model,
            cache_dir=DIFFUSERS_CACHE,
            resume_download=True,
            force_download=False,
            use_auth_token=token
        )
        folder_names = [k for k in config_dict.keys() if not k.startswith("_")]
        allow_patterns = [os.path.join(k, "*") for k in folder_names]
        allow_patterns += [WEIGHTS_NAME, SCHEDULER_CONFIG_NAME, CONFIG_NAME, ONNX_WEIGHTS_NAME, StableDiffusionPipeline.config_name]
    except Exception as e:
        print(e)
        allow_patterns = None
    
    # make sure we don't download flax, safetensors, or ckpt weights.
    ignore_patterns = ["*.msgpack", "*.safetensors", "*.ckpt"]
    try:
        _snapshot_download.snapshot_download(
            model,
            cache_dir=DIFFUSERS_CACHE,
            token=token,
            revision=revision,
            resume_download=True,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns
        )
    except utils._errors.RevisionNotFoundError:
        _snapshot_download.snapshot_download(
            model,
            cache_dir=DIFFUSERS_CACHE,
            token=token,
            resume_download=True,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns
        )

    future.set_done()