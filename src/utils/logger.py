"""Einheitliches Log-Format für alle Einstiegspunkte (Training, API, Monitoring)."""

import logging


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
