import shutil


def add_line_separator(s: str, n=None):
    if n is None:
        n = shutil.get_terminal_size().columns
    line_separator = "-" * n
    return f"{line_separator}\n{s}\n{line_separator}"


def printls(s: str, n=None):
    """Print a line separator followed by the input string and another line separator."""
    print(add_line_separator(s, n=n))
