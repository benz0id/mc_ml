import os
from pathlib import Path
from typing import Dict, List, Tuple, Callable
from timeit import default_timer as timer

from analysis.helpers import get_date_time_filename, append_csv_line


def add(d, k, v):
    if k in d:
        d[k].append(v)
    else:
        d[k] = [v]


class PipelineTimer:

    """
    Allows for convenient calculation and presentation of multistep pipeline
    timings..
    """

    epoch_to_step_times: Dict[int, List[Tuple[str, float]]]
    get_current_epoch: Callable[[], int]

    header: List[str]

    active: bool
    current_step: str | None
    start_time: float | None

    out_file: Path

    do_timing: bool
    verbose: bool

    def __init__(self, out_dir: Path,
                 pipeline_name: str,
                 get_current_epoch: Callable[[], int],
                 verbose: bool,
                 do_timing: bool = True):
        self.do_timing = do_timing
        self.verbose = verbose
        self.get_current_epoch = get_current_epoch

        if not self.do_timing:
            return
        self.last_rec = timer()
        self.epoch_to_step_times = {}
        self.out_file = (out_dir /
                         get_date_time_filename() /
                         (pipeline_name + '.csv'))
        if not self.out_file.parent.exists():
            self.out_file.parent.mkdir(exist_ok=True)
        self.active = False
        self.header = []

    def iterate(self):
        if not self.do_timing:
            return

        if not self.out_file.parent.exists():
            os.mkdir(self.out_file.parent)

        epoch = self.get_current_epoch()

        if not self.header:
            step_names = [
                step_time[0]
                for step_time in self.epoch_to_step_times[epoch]
            ]
            self.header = ['epoch'] + step_names
            append_csv_line(self.out_file, self.header)
        step_times = [('epoch', epoch)] + \
            self.epoch_to_step_times[epoch]
        data = []
        for i, step_time in enumerate(step_times):
            step, time = step_time
            #  Make sure that we are adding the value to the right column.
            if not step == self.header[i]:
                assert False
            data.append(time)
        append_csv_line(self.out_file, data)
        if self.verbose:
            print(self.get_iter_string(self.get_current_epoch()))
        self.active = False

    def get_iter_string(self, epoch: int):
        step_times = self.epoch_to_step_times[epoch]
        s = ''
        for step, time in step_times:
            s += f'{time: 3.3f}    \t{step}\n'
        return s

    def begin(self, step_name: str) -> None:
        if not self.do_timing:
            return

        if self.active:
            finish_time = timer()
            record = self.current_step, finish_time - self.start_time
            add(self.epoch_to_step_times, self.get_current_epoch(), record)
        self.active = True
        self.current_step = step_name
        self.start_time = timer()

    def end(self) -> None:
        if not self.do_timing:
            return

        finish_time = timer()
        record = self.current_step, finish_time - self.start_time
        add(self.epoch_to_step_times, self.get_current_epoch(), record)

        if self.verbose:
            print(finish_time - self.start_time)

        self.active = False
        self.current_step = None
        self.start_time = None
