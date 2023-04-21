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
    except:
        allow_patterns = None
    
    # make sure we don't download flax, safetensors, or ckpt weights.
    ignore_patterns = ["*.msgpack", "*.safetensors", "*.ckpt"]

    requested_pipeline_class = config_dict.get(
        "_class_name", StableDiffusionPipeline.__name__
    )
    user_agent = {"pipeline_class": requested_pipeline_class}
    user_agent = http_user_agent(user_agent)

    # download all allow_patterns

    # NOTE: Modified to yield the progress as an int from 0-100.
    def http_get(
        url: str,
        temp_file: BinaryIO,
        *,
        proxies=None,
        resume_size=0,
        headers: Optional[Dict[str, str]] = None,
        timeout=10.0,
        max_retries=0,
    ):
        headers = copy.deepcopy(headers)
        if resume_size > 0:
            headers["Range"] = "bytes=%d-" % (resume_size,)
        r = _request_wrapper(
            method="GET",
            url=url,
            stream=True,
            proxies=proxies,
            headers=headers,
            timeout=timeout,
            max_retries=max_retries,
        )
        hf_raise_for_status(r)
        content_length = r.headers.get("Content-Length")
        total = resume_size + int(content_length) if content_length is not None else 0
        progress = 0
        previous_value = 0
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                progress += len(chunk)
                value = progress / total
                if value - previous_value > 0.01:
                    previous_value = value
                    yield value
                temp_file.write(chunk)

    def hf_hub_download(
        repo_id: str,
        filename: str,
        *,
        subfolder: Optional[str] = None,
        repo_type: Optional[str] = None,
        revision: Optional[str] = None,
        library_name: Optional[str] = None,
        library_version: Optional[str] = None,
        cache_dir: Union[str, Path, None] = None,
        user_agent: Union[Dict, str, None] = None,
        force_download: Optional[bool] = False,
        force_filename: Optional[str] = None,
        proxies: Optional[Dict] = None,
        etag_timeout: Optional[float] = 10,
        resume_download: Optional[bool] = False,
        use_auth_token: Union[bool, str, None] = None,
        local_files_only: Optional[bool] = False,
        legacy_cache_layout: Optional[bool] = False,
    ):
        if force_filename is not None:
            warnings.warn(
                "The `force_filename` parameter is deprecated as a new caching system, "
                "which keeps the filenames as they are on the Hub, is now in place.",
                FutureWarning,
            )
            legacy_cache_layout = True

        if legacy_cache_layout:
            url = hf_hub_url(
                repo_id,
                filename,
                subfolder=subfolder,
                repo_type=repo_type,
                revision=revision,
            )

            return cached_download(
                url,
                library_name=library_name,
                library_version=library_version,
                cache_dir=cache_dir,
                user_agent=user_agent,
                force_download=force_download,
                force_filename=force_filename,
                proxies=proxies,
                etag_timeout=etag_timeout,
                resume_download=resume_download,
                use_auth_token=use_auth_token,
                local_files_only=local_files_only,
                legacy_cache_layout=legacy_cache_layout,
            )

        if cache_dir is None:
            cache_dir = HUGGINGFACE_HUB_CACHE
        if revision is None:
            revision = DEFAULT_REVISION
        if isinstance(cache_dir, str):
            cache_dir = Path(cache_dir)

        url_name = filename

        if not subfolder or subfolder == "":
            # This is used to create a URL, and not a local path, hence the forward slash.
            url_name = f"{subfolder}/{filename}"
            subfolder = ""

        if repo_type is None:
            repo_type = "model"
        if repo_type not in REPO_TYPES:
            raise ValueError(
                f"Invalid repo type: {repo_type}. Accepted repo types are:"
                f" {str(REPO_TYPES)}"
            )

        storage_folder = cache_dir / repo_folder_name(
            repo_id=repo_id, repo_type=repo_type
        )
        storage_folder.mkdir(parents=True, exist_ok=True)

        # if user provides a commit_hash and they already have the file on disk,
        # shortcut everything.
        if REGEX_COMMIT_HASH.match(revision):
            pointer_path = (
                storage_folder / "snapshots" / revision / subfolder / filename
            )

            if (pointer_path).exists():
                return pointer_path.as_posix()

        url = hf_hub_url(repo_id, url_name, repo_type=repo_type, revision=revision)

        headers = build_hf_headers(
            use_auth_token=use_auth_token,
            library_name=library_name,
            library_version=library_version,
            user_agent=user_agent,
        )

        url_to_download = url
        etag = None
        commit_hash = None
        if not local_files_only:
            try:
                try:
                    metadata = get_hf_file_metadata(
                        url=url,
                        use_auth_token=use_auth_token,
                        proxies=proxies,
                        timeout=etag_timeout,
                    )
                except EntryNotFoundError as http_error:
                    # Cache the non-existence of the file and raise
                    commit_hash = http_error.response.headers.get(
                        HUGGINGFACE_HEADER_X_REPO_COMMIT
                    )
                    if commit_hash is not None and not legacy_cache_layout:
                        no_exist_file_path = (
                            storage_folder
                            / ".no_exist"
                            / commit_hash
                            / subfolder
                            / filename
                        )
                        no_exist_file_path.parent.mkdir(parents=True, exist_ok=True)
                        no_exist_file_path.touch()
                        _cache_commit_hash_for_specific_revision(
                            storage_folder, revision, commit_hash
                        )
                    raise

                # Commit hash must exist
                commit_hash = metadata.commit_hash
                if commit_hash is None:
                    raise OSError(
                        "Distant resource does not seem to be on huggingface.co (missing"
                        " commit header)."
                    )

                # Etag must exist
                etag = metadata.etag
                # We favor a custom header indicating the etag of the linked resource, and
                # we fallback to the regular etag header.
                # If we don't have any of those, raise an error.
                if etag is None:
                    raise OSError(
                        "Distant resource does not have an ETag, we won't be able to"
                        " reliably ensure reproducibility."
                    )

                # In case of a redirect, save an extra redirect on the request.get call,
                # and ensure we download the exact atomic version even if it changed
                # between the HEAD and the GET (unlikely, but hey).
                # Useful for lfs blobs that are stored on a CDN.
                if metadata.location != url:
                    url_to_download = metadata.location
                    if (
                        "lfs.huggingface.co" in url_to_download
                        or "lfs-staging.huggingface.co" in url_to_download
                    ):
                        # Remove authorization header when downloading a LFS blob
                        headers.pop("authorization", None)
            except (requests.exceptions.SSLError, requests.exceptions.ProxyError):
                # Actually raise for those subclasses of ConnectionError
                raise
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                OfflineModeIsEnabled,
            ):
                # Otherwise, our Internet connection is down.
                # etag is None
                pass

        # etag is None == we don't have a connection or we passed local_files_only.
        # try to get the last downloaded one from the specified revision.
        # If the specified revision is a commit hash, look inside "snapshots".
        # If the specified revision is a branch or tag, look inside "refs".
        if etag is None:
            # In those cases, we cannot force download.
            if force_download:
                raise ValueError(
                    "We have no connection or you passed local_files_only, so"
                    " force_download is not an accepted option."
                )
            if REGEX_COMMIT_HASH.match(revision):
                commit_hash = revision
            else:
                ref_path = storage_folder / "refs" / revision
                commit_hash = ref_path.read_text()

            pointer_path = (
                storage_folder / "snapshots" / commit_hash / subfolder / filename
            )
            if pointer_path.exists():
                return pointer_path.as_posix()

            # If we couldn't find an appropriate file on disk,
            # raise an error.
            # If files cannot be found and local_files_only=True,
            # the models might've been found if local_files_only=False
            # Notify the user about that
            if local_files_only:
                raise LocalEntryNotFoundError(
                    "Cannot find the requested files in the disk cache and"
                    " outgoing traffic has been disabled. To enable hf.co look-ups"
                    " and downloads online, set 'local_files_only' to False."
                )
            else:
                raise LocalEntryNotFoundError(
                    "Connection error, and we cannot find the requested files in"
                    " the disk cache. Please try again or make sure your Internet"
                    " connection is on."
                )

        # From now on, etag and commit_hash are not None.
        blob_path = storage_folder / "blobs" / etag
        pointer_path = storage_folder / "snapshots" / commit_hash / subfolder / filename
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        pointer_path.parent.mkdir(parents=True, exist_ok=True)

        # if passed revision is not identical to commit_hash
        # then revision has to be a branch name or tag name.
        # In that case store a ref.
        _cache_commit_hash_for_specific_revision(
            storage_folder.as_posix(), revision, commit_hash
        )

        if pointer_path.exists() and not force_download:
            return pointer_path.as_posix()

        if blob_path.exists() and not force_download:
            # we have the blob already, but not the pointer
            logger.info("creating pointer to %s from %s", blob_path, pointer_path)
            _create_relative_symlink(
                blob_path.as_posix(), pointer_path.as_posix(), new_blob=False
            )
            return pointer_path.as_posix()

        # Prevent parallel downloads of the same file with a lock.
        lock_path = blob_path.with_suffix(".lock")

        with FileLock(lock_path):
            # If the download just completed while the lock was activated.
            if pointer_path.exists() and not force_download:
                # Even if returning early like here, the lock will be released.
                return pointer_path.as_posix()

            if resume_download:
                incomplete_path = blob_path.with_suffix(".incomplete")

                @contextmanager
                def _resumable_file_manager() -> "Generator[io.BufferedWriter,None,None]":
                    with open(incomplete_path, "ab") as f:
                        yield f

                temp_file_manager = _resumable_file_manager
                resume_size = (
                    incomplete_path.stat().st_size if incomplete_path.exists() else 0
                )
            else:
                temp_file_manager = partial(
                    tempfile.NamedTemporaryFile, mode="wb", dir=cache_dir, delete=False
                )
                resume_size = 0

            # Download to temporary file, then copy to cache dir once finished.
            # Otherwise you get corrupt cache entries if the download gets interrupted.
            with temp_file_manager() as temp_file:
                logger.info("downloading %s to %s", url, temp_file.name)

                yield from http_get(
                    url_to_download,
                    temp_file,
                    proxies=proxies,
                    resume_size=resume_size,
                    headers=headers,
                )

            logger.info("storing %s in cache at %s", url, blob_path)
            Path(temp_file.name).replace(blob_path)

            logger.info("creating pointer to %s from %s", blob_path, pointer_path)
            _create_relative_symlink(
                blob_path.as_posix(), pointer_path.as_posix(), new_blob=True
            )

        with contextlib.suppress(OSError):
            lock_path.unlink()

    @validate_hf_hub_args
    def snapshot_download(
        repo_id: str,
        *,
        revision: Optional[str] = None,
        repo_type: Optional[str] = None,
        cache_dir: Union[str, Path, None] = None,
        library_name: Optional[str] = None,
        library_version: Optional[str] = None,
        user_agent: Optional[Union[Dict, str]] = None,
        proxies: Optional[Dict] = None,
        etag_timeout: Optional[float] = 10,
        resume_download: Optional[bool] = False,
        use_auth_token: Optional[Union[bool, str]] = None,
        local_files_only: Optional[bool] = False,
        allow_regex: Optional[Union[List[str], str]] = None,
        ignore_regex: Optional[Union[List[str], str]] = None,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
    ):
        if cache_dir is None:
            cache_dir = HUGGINGFACE_HUB_CACHE
        if revision is None:
            revision = DEFAULT_REVISION
        if isinstance(cache_dir, Path):
            cache_dir = str(cache_dir)

        if repo_type is None:
            repo_type = "model"
        if repo_type not in REPO_TYPES:
            raise ValueError(
                f"Invalid repo type: {repo_type}. Accepted repo types are:"
                f" {str(REPO_TYPES)}"
            )

        storage_folder = cache_dir / repo_folder_name(
            repo_id=repo_id, repo_type=repo_type
        )

        # TODO: remove these 4 lines in version 0.12
        #       Deprecated code to ensure backward compatibility.
        if allow_regex is not None:
            allow_patterns = allow_regex
        if ignore_regex is not None:
            ignore_patterns = ignore_regex

        # if we have no internet connection we will look for an
        # appropriate folder in the cache
        # If the specified revision is a commit hash, look inside "snapshots".
        # If the specified revision is a branch or tag, look inside "refs".
        if local_files_only:
            if REGEX_COMMIT_HASH.match(revision):
                commit_hash = revision
            else:
                # retrieve commit_hash from file
                ref_path = storage_folder / "refs" / revision
                commit_hash = ref_path.read_text()

            snapshot_folder = storage_folder / "snapshots" / commit_hash

            if snapshot_folder:
                return snapshot_folder.as_posix()

            raise ValueError(
                "Cannot find an appropriate cached snapshot folder for the specified"
                " revision on the local disk and outgoing traffic has been disabled. To"
                " enable repo look-ups and downloads online, set 'local_files_only' to"
                " False."
            )

        # if we have internet connection we retrieve the correct folder name from the huggingface api
        _api = HfApi()
        repo_info = _api.repo_info(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            use_auth_token=use_auth_token,
        )
        filtered_repo_files = list(
            filter_repo_objects(
                items=[f.rfilename for f in repo_info.siblings],
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
        )
        commit_hash = repo_info.sha
        snapshot_folder = storage_folder / "snapshots" / commit_hash
        # if passed revision is not identical to commit_hash
        # then revision has to be a branch name or tag name.
        # In that case store a ref.
        if revision != commit_hash:
            ref_path = storage_folder / "refs" / revision
            ref_path.mkdir(parents=True, exist_ok=True)
            with open(ref_path, "w") as f:
                f.write(commit_hash)

        # we pass the commit_hash to hf_hub_download
        # so no network call happens if we already
        # have the file locally.

        for i, repo_file in tqdm(
            enumerate(filtered_repo_files), f"Fetching {len(filtered_repo_files)} files"
        ):
            yield DownloadStatus(repo_file, i, len(filtered_repo_files))
            for status in hf_hub_download(
                repo_id,
                filename=repo_file,
                repo_type=repo_type,
                revision=commit_hash,
                cache_dir=cache_dir,
                library_name=library_name,
                library_version=library_version,
                user_agent=user_agent,
                proxies=proxies,
                etag_timeout=etag_timeout,
                resume_download=resume_download,
                use_auth_token=use_auth_token,
            ):
                yield DownloadStatus(repo_file, int(status), 1)
            yield DownloadStatus(repo_file, i + 1, len(filtered_repo_files))

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