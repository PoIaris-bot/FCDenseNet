import os
import cv2
import numpy as np
from collections import defaultdict


def increment_path(path):
    if os.path.exists(path):
        for n in range(2, 9999):
            temp_path = f'{path}{n}'
            if not os.path.exists(temp_path):
                return temp_path
    return path


def localization(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = []
    for i in range(len(contours)):
        area.append(cv2.contourArea(contours[i]))
    max_idx = np.argmax(area)
    m = cv2.moments(contours[max_idx])
    x = int(m['m10'] / m['m00'])
    y = int(m['m01'] / m['m00'])
    return (x, y), [contours[max_idx]]


class MetricMonitor:
    def __init__(self, float_precision=3):
        self.float_precision = float_precision
        self.reset()

    def reset(self):
        self.metrics = defaultdict(lambda: {'val': 0, 'count': 0, 'avg': 0})

    def update(self, metric_name, val):
        metric = self.metrics[metric_name]

        metric['val'] += val
        metric['count'] += 1
        metric['avg'] = metric['val'] / metric['count']

    def __str__(self):
        return ' | '.join(
            [
                '{metric_name}: {avg:.{float_precision}f}'.format(
                    metric_name=metric_name, avg=metric['avg'], float_precision=self.float_precision
                )
                for (metric_name, metric) in self.metrics.items()
            ]
        )
