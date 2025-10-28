from abc import ABC, abstractmethod

class BaseModel(ABC):
    @abstractmethod
    def encode(self, texts):
        raise NotImplementedError()

    @abstractmethod
    def retrieve(self, query, corpus, top_k):
        raise NotImplementedError()
