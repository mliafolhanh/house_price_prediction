import coloredlogs  # noqa
import logging
import logging.config


def get_logging_config():
    DEFAULT_LOGGING = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "colored": {
                "()": "coloredlogs.ColoredFormatter",
                "format": "%(asctime)s %(process)d %(threadName)10s %(filename)15s:%(lineno)3d %(levelname)5s--: [%(name)s] %(message)s"  # noqa
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "colored",
                "level": "INFO",
            },
        },
        "root": {
            "level": "INFO",
            "handlers": ["console"],
            "propagate": "yes"
        }
    }

    return DEFAULT_LOGGING


def setup_default_logging():
    logging.config.dictConfig(get_logging_config())
