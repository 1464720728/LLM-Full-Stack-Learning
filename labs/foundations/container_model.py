"""Overlay and volume lifetimes, deliberately simplified."""
from dataclasses import dataclass, field

@dataclass
class Container:
    image: dict
    writable: dict = field(default_factory=dict)
    volume: dict = field(default_factory=dict)
    running: bool = True

    def read(self, path):
        if path.startswith("/data/"):
            return self.volume[path]
        return self.writable[path] if path in self.writable else self.image[path]

    def write(self, path, value):
        target = self.volume if path.startswith("/data/") else self.writable
        target[path] = value

if __name__ == "__main__":
    image, volume = {"/app/version": "v1"}, {}
    first = Container(image, volume=volume)
    first.write("/app/version", "temporary-v2")
    first.write("/data/progress", "lesson13")
    second = Container(image, volume=volume)
    print("new container app:", second.read("/app/version"))
    print("persistent volume:", second.read("/data/progress"))
