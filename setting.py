from dataclasses import dataclass
import logging
import sys


@dataclass(frozen=True)
class Setting:
    URL_GET = 'match/'
    URL_POST = 'match/make_match/'
    API_VERSION = 'v1'
    PORT = '8000'
    HOST = 'localhost'
    START_TIME = '23:00'


def create_logger():
    stream_handler = logging.StreamHandler(sys.stdout)
    logging.basicConfig(
        handlers=[stream_handler,],
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.WARNING,
    )
    return logging.getLogger(__name__)
