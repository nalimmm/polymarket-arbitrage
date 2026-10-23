"""Run the bot in paper mode:  python main.py [--config config.yaml]"""

import argparse
import asyncio
import logging

from bot.config import load_config
from bot.engine import Engine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    cfg = load_config(args.config)
    if cfg.execution.mode != "paper":
        raise SystemExit("v2 only supports paper mode")
    try:
        asyncio.run(Engine(cfg).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
