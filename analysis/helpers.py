from datetime import datetime
from pathlib import Path
from typing import List, Any

RUN_DATETIME = None


def append_csv_line(filepath: Path, line: List[Any]):
    with open(filepath, 'a') as outfile:
        outline = []
        for val in line:
            if isinstance(val, float):
                outline.append(f"{val:.3f}")
            else:
                outline.append(str(val))
        s = ', '.join(outline + ['\n'])
        outfile.write(s)


def get_date_time_filename() -> str:
    global RUN_DATETIME
    if not RUN_DATETIME:
        RUN_DATETIME = datetime.now()
    return RUN_DATETIME.strftime("%Y-%m-%d-%H-%M-%S")
