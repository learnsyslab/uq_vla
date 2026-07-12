"""Worker entrypoint for evaluating one ensemble member on a dedicated device."""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
from pathlib import Path
import sys

from lerobot.utils.import_utils import register_third_party_plugins

from .evaluation import evaluate_member_request, load_member_evaluation_request

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request_path", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    #mp.set_start_method("spawn", force=True)
    args = _parse_args()
    request = load_member_evaluation_request(args.request_path)

    try:
        register_third_party_plugins()
        evaluation = evaluate_member_request(request)
    except Exception:
        logger.exception(
            "Evaluation worker crashed for round %s member %s",
            request.round_index,
            request.member_index,
        )
        raise

    logger.info(
        "Finished evaluation for round %s member %s -> %s",
        evaluation.round_index,
        evaluation.member_index,
        request.result_path,
    )


if __name__ == "__main__":
    main()
    