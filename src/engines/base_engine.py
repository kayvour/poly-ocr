from abc import ABC, abstractmethod


class BaseOCREngine(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def predict(self, image_path: str, lang: str = "eng"):
        """
        Returns:
            text (str)
            confidence (float or None)
            inference_time (float)
        """
        pass
        