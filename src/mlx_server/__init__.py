from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("python-mlx-server")
except PackageNotFoundError:
    __version__ = "0.2.0"
