import logging

import uvicorn

from mlx_server.config import settings


def main() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("mlx_server")
    logger.info(f"Starting MLX Server on {settings.host}:{settings.port}")
    logger.info(f"Registry: {settings.model_registry_path}")

    uvicorn.run(
        "mlx_server.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
