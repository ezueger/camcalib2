from .source import FrameSource, ImageFolderSource, VideoFileSource

__all__ = ["FrameSource", "ImageFolderSource", "VideoFileSource", "GenICamSource"]


def __getattr__(name):
    # lazy import: harvesters is an optional dependency
    if name == "GenICamSource":
        from .genicam import GenICamSource
        return GenICamSource
    raise AttributeError(name)
