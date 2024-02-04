import pandas as pd
import csv
from pathlib import Path

from FlexML.Sample import Sample, SampleCollection


class MetadataSample(Sample):
    """Sample defined by a metadata vector."""

    def get_metadata(self) -> list:
        raise NotImplementedError()

class MetadataSampleCollection(SampleCollection):
    """Dataset defined by a folder which contains a 'metadata.csv' where each row corresponds to a MetadataSample."""

    def __init__(self, path: Path):
        super().__init__(path)

    def from_metadata(self, metadata) -> MetadataSample:
        raise NotImplementedError()

    def get_metadata_headers(self) -> list[str]:
        raise NotImplementedError()
    
    def publish_all_samples(self):
        self.all_metadata = pd.read_csv(self.dataset_path / "metadata.csv")
        for i in range(len(self.all_metadata)):
            sample = self.from_metadata(self.all_metadata.iloc[i])
            self.new_sample.emit(sample)
    
    def add_sample(self, sample: MetadataSample):
        metadata_path = self.dataset_path / "metadata.csv"
        file_exists = metadata_path.is_file()
        with open(metadata_path, "a+", newline="", encoding="UTF8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(self.get_metadata_headers())
            writer.writerow(sample.get_metadata())
    
        self.new_sample.emit(sample)
