import logging


def setup_logging(
    level: int = logging.INFO,
):
    """
    Configure application-wide logging.
    """

    logging.basicConfig(
        level=level,
        format=(
            "%(asctime)s | "
            "%(levelname)s | "
            "%(name)s | "
            "%(message)s"
        ),
    )