from pydantic import BaseModel
import json


class SerializableModel(BaseModel):
    def to_file(self, file_path: str):
        with open(file_path, "w") as f:
            json.dump(self.dict(), f, indent=4)

    @classmethod
    def from_file(cls, file_path: str):
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(**data)
