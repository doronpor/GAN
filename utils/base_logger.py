import logging
import sys


def set_base_logger():
    handler = [logging.StreamHandler(sys.stdout)]
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=handler)
