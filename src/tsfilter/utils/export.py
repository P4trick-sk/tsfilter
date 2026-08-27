from typing import List

from tsfilter.dataset.sample import DatasetSample

import csv

def save_as_csv(samples: List[DatasetSample], path: str):
    with open(path, 'w', newline='') as file:
        writer = csv.writer(file)
        for sample in samples:
            writer.writerow([sample.clean.values, sample.noisy.values])