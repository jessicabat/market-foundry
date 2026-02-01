import re

def normalize_text(s: str) -> str:
    # common weird chars
    s = s.replace("\x00", "")
    s = s.replace("\xa0", " ")

    # unify newlines
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # collapse spaces/tabs
    s = re.sub(r"[ \t]+", " ", s)

    # collapse huge blank runs
    s = re.sub(r"\n{3,}", "\n\n", s)

    # remove space before newlines
    s = re.sub(r" +\n", "\n", s)

    return s.strip()
