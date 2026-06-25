import os
import json
from abc import ABC, abstractmethod
from pathlib import Path


import io
import numpy as np
import jax.numpy as jnp
from PIL.Image import Image as PILImage
from PIL.Image import fromarray as pil_fromarray, open as pil_open

class BaseLoggerWrapper(ABC):
    """Abstract base class for logger wrappers."""

    @abstractmethod
    def __init__(self, *args, **kwargs):
        """Initialize the logger wrapper."""
        raise NotImplementedError

    @abstractmethod
    def log(self, data: dict, **kwargs):
        """Log data to the backend."""
        raise NotImplementedError

    @abstractmethod
    def finish(self, **kwargs):
        """Finish logging and cleanup resources."""
        raise NotImplementedError

    @abstractmethod
    def Image(self, *args, **kwargs):
        """Format an image for the backend."""
        raise NotImplementedError


class TrackIOWrapper(BaseLoggerWrapper):
    """Logger wrapper for TrackIO experiment tracking."""

    def __init__(
        self,
        project: str,
        name: str | None = None,
        space_id: str | None = None,
        dataset_id: str | None = None,
        config: dict | None = None,
        resume: str = "never",
        **kwargs,
    ):
        """Initialize TrackIO logging session."""
        import trackio

        self.writer = trackio.init(
            project,
            name=name,
            space_id=space_id,
            dataset_id=dataset_id,
            config=config,
            resume=resume,
        )

    def log(self, data: dict, **kwargs):
        """Log data to TrackIO."""
        self.writer.log(data)

    def finish(self, **kwargs):
        """Finish TrackIO session."""
        self.writer.finish()

    def Image(self, *args, **kwargs):
        """TrackIO does not support image logging."""
        raise NotImplementedError("TrackIO does not support image logging directly.")


class WandBWrapper(BaseLoggerWrapper):
    """Logger wrapper for Weights & Biases."""

    def __init__(self, *args, **kwargs):
        """Initialize W&B logging session."""
        import wandb

        wandb.init(*args, **kwargs)
        self.writer = wandb

    def log(self, data: dict, step: int | None = None, commit: bool | None = None, **kwargs):
        """Log data to W&B."""
        self.writer.log(data, step=step, commit=commit)

    def finish(self, exit_code: int | None = None, **kwargs):
        """Finish W&B session."""
        self.writer.finish(exit_code=exit_code)

    def Image(self, *args, **kwargs):
        """Create W&B Image object."""
        return self.writer.Image(*args, **kwargs)
    
class CometMLWrapper(BaseLoggerWrapper):
    """Logger wrapper for Comet ML."""

    def __init__(self, *args, **kwargs):
        """Initialize Comet ML logging session."""
        import comet_ml
        project_name = kwargs.pop("project", None)
        experiment_name = kwargs.pop("experiment_name", None)
        workspace = kwargs.pop("entity", None)
        offline_directory = kwargs.pop("offline_directory", "./comet_offline_logs")
        config = kwargs.pop("config", {})

        api_key = os.environ.get("COMET_API_KEY")
        offline_mode = os.environ.get("COMET_OFFLINE", "0") == "1"

        if offline_mode:
            os.makedirs(offline_directory, exist_ok=True)
            self.experiment = comet_ml.OfflineExperiment(
                project_name=project_name,
                experiment_name=experiment_name,
                workspace=workspace,
                offline_directory=offline_directory,
            )
            print(f"[CometML] Running in offline mode. Logs will be saved to: {offline_directory}")
        else:
            self.experiment = comet_ml.Experiment(
                project_name=project_name,
                experiment_name=experiment_name,
                workspace=workspace,
                api_key=api_key,
                *args,
                **kwargs
            )

        if config:
            self.experiment.log_parameters(config)

    def log(self, data: dict, step: int | None = None, commit: bool | None = None, **kwargs):
        self.experiment.log_metrics(data, step=step)

    def finish(self, exit_code: int | None = None, **kwargs):
        """Finish Comet ML session."""
        self.experiment.end()

    def Image(self, image, caption=None, step = 0, **kwargs):
        """Log image to Comet ML and return the PIL Image for compatibility."""
        if isinstance(image, np.ndarray):
            pil_image = pil_fromarray(image.astype(np.uint8))
        elif isinstance(image, jnp.ndarray):
            pil_image = pil_fromarray(np.array(image).astype(np.uint8))
        else:
            pil_image = image

        buf = io.BytesIO()
        pil_image.save(buf, format='PNG')
        buf.seek(0)

        self.experiment.log_image(
            image_data=buf,
            name=caption if caption is not None else 'image',
            **kwargs
        )

        return pil_image

class Writer:
    """Unified experiment logging interface supporting multiple backends."""

    SUPPORTED_TYPES = ("wandb", "trackio", "comet_ml", None)

    def init(
        self,
        writer_type: str | None = None,
        save_locally: bool = False,
        log_dir: str = "logs/",
        config: dict | None = None,
        **kwargs,
    ):
        """Initialize Writer with backend and local storage configuration.

        Args:
            writer_type: Backend type ("wandb", "trackio", or None)
            save_locally: Whether to save logs locally
            log_dir: Path to the log directory (default: "logs/")
            config: Configuration dict to save and pass to backend
            **kwargs: Backend-specific initialization arguments
        """
        assert writer_type in self.SUPPORTED_TYPES, (
            f"Writer type {writer_type} is not supported. "
            f"Supported types are: {self.SUPPORTED_TYPES}."
        )

        if writer_type == "wandb":
            self.backend = WandBWrapper(config=config, **kwargs)
        elif writer_type == "trackio":
            self.backend = TrackIOWrapper(config=config, **kwargs)
        elif writer_type == "comet_ml":
            self.backend = CometMLWrapper(config = config, **kwargs)
        elif writer_type is None:
            self.backend = None

        self._step = 0
        self._image_counter = 0

        self.save_locally = save_locally
        self.log_dir = log_dir

        if self.save_locally:
            os.makedirs(self.log_dir, exist_ok=True)
            self._save_json("config.json", config)

    def log(self, data: dict, step=None, **kwargs):
        """Log data to both local storage and backend if configured.

        Args:
            data: Dictionary of metrics and values to log
            **kwargs: Additional arguments for backend logging
        """
        if self.save_locally:
            self._save_logs(data, step)

        if self.backend:
            self.backend.log(data, step=step, **kwargs)

    def Image(self, data_or_path: str | Path | np.ndarray | PILImage, **kwargs):
        """Create image object for logging from various input formats.

        Args:
            data_or_path: Image data (file path, numpy array, or PIL Image)
            **kwargs: Additional arguments for backend Image creation

        Returns:
            Backend-specific image object or PIL Image if no backend
        """
        image = self._load_image(data_or_path)
        return self.backend.Image(image, **kwargs) if self.backend else image

    def finish(self, *args, **kwargs):
        """Finish logging session and cleanup resources."""
        if self.backend:
            self.backend.finish(*args, **kwargs)

    def _save_logs(self, data: dict, step=None):
        """Save data to local JSONL file with step counter."""
        save_dict = {}
        for key, value in data.items():
            if jnp.isscalar(value):
                save_dict[key] = float(value)
            elif isinstance(value, PILImage):
                save_dict[key] = self._save_image(value)
            elif hasattr(value, "image"):
                save_dict[key] = self._save_image(value.image)

        if step is not None:
            save_dict["step"] = int(step)
        else:
            save_dict["step"] = self._step
            self._step += 1

        log_file = os.path.join(self.log_dir, "log.jsonl")
        with open(log_file, "a", encoding="utf-8") as f:
            json.dump(save_dict, f, ensure_ascii=False)
            f.write("\n")

    def _save_image(self, image: PILImage):
        """Save PIL Image to local directory with sequential naming.

        Args:
            image: PIL Image to save

        Returns:
            str: Path to saved image file
        """

        image_path = os.path.join(self.log_dir, "images/", f"image_{self._image_counter:04d}.png")
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        self._image_counter += 1
        image.save(image_path)

        return image_path

    def _save_json(self, name: str, data: dict):
        """Save dictionary as JSON file in log directory.

        Args:
            name: Filename for JSON file
            data: Dictionary to save
        """

        json_path = os.path.join(self.log_dir, name)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def _load_image(self, data_or_path: str | Path | np.ndarray | PILImage) -> PILImage:
        """Load image from file path, numpy array, or PIL Image.

        Args:
            data_or_path: Image data in supported format

        Returns:
            PIL Image object

        Raises:
            ValueError: If input type is not supported
        """

        if isinstance(data_or_path, (str, Path)):
            image = pil_open(data_or_path)
        elif isinstance(data_or_path, np.ndarray):
            image = pil_fromarray(data_or_path.astype(np.uint8))
        elif isinstance(data_or_path, PILImage):
            image = data_or_path
        else:
            raise ValueError("Unsupported image data type.")

        return image