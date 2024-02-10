import pandas as pd
from pathlib import Path
from PyQt6.QtCore import pyqtSlot

from FlexML.Sample import Sample, SampleCollection


class MetadataSample(Sample):
    """Sample defined by a metadata vector."""

    def get_metadata(self) -> list:
        raise NotImplementedError()


class MetadataSampleCollection(SampleCollection):
    """SampleCollection defined by a folder which contains a 'metadata.csv' where each row corresponds to a MetadataSample."""

    def __init__(self, path: Path):
        super().__init__(path)
        self.metadata_csv_path = self.dataset_path / "metadata.csv"

    def from_metadata(self, metadata) -> MetadataSample:
        """Returns a MetadataSample created from a metadata vector."""

        raise NotImplementedError()

    def get_metadata_headers(self) -> list[str]:
        """Returns the CSV headers associated with the metadata vectors of the collection."""

        raise NotImplementedError()
    
    def publish_all_samples(self) -> int:
        all_metadata = pd.read_csv(self.metadata_csv_path)
        for _, row in all_metadata.iterrows():
            sample = self.from_metadata(row)
            self.new_sample.emit(sample)
        return len(all_metadata)
    
    @pyqtSlot(MetadataSample)
    def add_sample(self, sample: MetadataSample):
        df = pd.DataFrame([sample.get_metadata()], columns=self.get_metadata_headers())
        df.to_csv(self.metadata_csv_path, mode='a', header=not self.metadata_csv_path.is_file(), index=False)
        self.new_sample.emit(sample)
