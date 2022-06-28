import glob
import tensorflow_datasets as tfds

class MyDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for my_dataset dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict(
                {
                    "image": tfds.features.Image(shape=(256, 256, 3)),
                    "label": tfds.features.ClassLabel(
                        names=["cat", "dog"]
                    ),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        """Download the data and define splits."""
        #extracted_path = dl_manager.download_and_extract("http://data.org/data.zip")
        
        # dl_manager returns pathlib-like objects with `path.read_text()`,
        # `path.iterdir()`,...
        extracted_path = 'D:\\data\\Cats_and_Dogs\\dataset'
        return {
            "train": self._generate_examples(path=extracted_path+"\\training_set"),
            "test": self._generate_examples(path=extracted_path +"\\test_set"),
        }

    def _generate_examples(self, path):  # -> Iterator[Tuple[Key, Example]]
        """Generator of examples for each split."""
        for img_path in glob.glob(path+"\\*\\*jpg"):
            # Yields (key, example)
            yield img_path, {
                "image": img_path,
                "label": "cat" if "cat" in path else "dog",
            }
