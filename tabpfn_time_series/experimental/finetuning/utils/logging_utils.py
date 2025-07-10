import logging
import sys
from pathlib import Path


class StreamToLogger:
    """
    A class to redirect a stream (like stdout or stderr) to a logger.
    It buffers writes until a newline is received.
    """

    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.level = level
        self.linebuf = ""

    def write(self, buf: str):
        # Treat \r as a newline, which is what progress bars use
        buf = buf.replace("\r", "\n")
        self.linebuf += buf
        lines = self.linebuf.split("\n")
        # The last item is the incomplete line, so we keep it in the buffer
        self.linebuf = lines.pop()
        for line in lines:
            # Avoid logging empty lines
            if line:
                self.logger.log(self.level, line)

    def flush(self):
        # When the stream is flushed, log anything remaining in the buffer.
        if self.linebuf:
            self.logger.log(self.level, self.linebuf)
            self.linebuf = ""


def configure_logging(
    output_dir: Path, run_name: str, should_log_to_file: bool = True
) -> None:
    """
    Configures the root logger to redirect stdout/stderr and optionally log to a file.
    This function should be called only once.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent conflicts or duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add a handler to print to the original console
    console_handler = logging.StreamHandler(sys.__stdout__)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Redirect stdout and stderr to the logging system
    sys.stdout = StreamToLogger(logging.getLogger("stdout"), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger("stderr"), logging.ERROR)

    if should_log_to_file:
        # Add a file handler to also save logs to a file in the output directory
        log_file = output_dir / f"{run_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)  # Use the same formatter
        root_logger.addHandler(file_handler)

    logging.info("Logging configured. All output will be captured.")
