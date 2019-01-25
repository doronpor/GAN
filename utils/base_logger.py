import logging
import sys
import os.path as path
import os


def set_base_logger(log_path=None, level=logging.INFO):
    if log_path is None:
        log_root = path.abspath(path.join(path.dirname(__file__), '../log/'))
        if not path.isdir(log_root):
            os.mkdir(log_root)
        log_path = path.join(log_root, 'train.log')

    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')

    logging.basicConfig(level=level,
                        format='%(levelname)s - %(name)s -  %(message)s',
                        handlers=[stream_handler, file_handler])
