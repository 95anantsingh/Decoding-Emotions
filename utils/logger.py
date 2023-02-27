"""
This file contains basic logging logic.
"""

import logging

LEVELS = {
    'critical': logging.CRITICAL, 'error': logging.ERROR, 'warning': logging.WARNING,
    'info': logging.INFO, 'debug': logging.DEBUG}

names = set()
no_fmt_logger = None

class DispatchingFormatter:
    """Dispatch formatter for logger and it's sub logger."""
    def __init__(self, formatters, default_formatter):
        self._formatters = formatters
        self._default_formatter = default_formatter

    def format(self, record):
        # Search from record's logger up to it's parents:
        logger = logging.getLogger(record.name)
        while logger:
            # Check if suitable formatter for current logger exists:
            if logger.name in self._formatters:
                formatter = self._formatters[logger.name]
                break
            else:
                logger = logger.parent
        else:
            # If no formatter found, just use default:
            formatter = self._default_formatter
        return formatter.format(record)


def __setup_custom_logger(name: str, level:str, logfile:str=None) -> logging.Logger:

    global no_fmt_logger

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    # formatter = logging.Formatter(fmt='%(levelname)s >> %(message)s')
    formatter = DispatchingFormatter({name+'_no_fmt_logger': logging.Formatter('')},
                    logging.Formatter('%(levelname)s >> %(message)s'))

    names.add(name)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    if logfile:
        file_handler = logging.FileHandler(logfile)
        file_handler.setFormatter(formatter)

    logger = logging.getLogger(name)

    try: logger.setLevel(LEVELS[level.lower()])
    except: logger.setLevel(logging.INFO)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    if no_fmt_logger == None:
        names.add(name+'_no_fmt_logger')
        no_fmt_logger = logging.getLogger(name+'_no_fmt_logger')
        no_fmt_logger.setLevel(logging.DEBUG)
        no_fmt_logger.addHandler(stream_handler)
        no_fmt_logger.addHandler(file_handler)

    return logger, no_fmt_logger


def get_logger(name:str, level:str, logfile:str=None) -> logging.Logger:
    if name in names:
        return logging.getLogger(name)
    else:
        return __setup_custom_logger(name,level,logfile)


class NoFmtLog:
    def __init__(self,logger) -> None:
        self.logger = logger

    def __call__(self,num_new_line:int=1,msg:str=''):
        if msg:
            self.logger.info(msg)
        else:
            for i in range(num_new_line):
                self.logger.info(msg)