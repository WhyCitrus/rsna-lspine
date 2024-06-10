import torchmetrics as tm

from .classification import *
from .segmentation import *
from .detection import *


class Dummy(tm.Metric):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def update(self, *args, **kwargs):
        pass

    def compute(self, *args, **kwargs):
        return {"val": 0}
