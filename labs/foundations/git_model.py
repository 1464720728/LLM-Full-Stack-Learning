"""A tiny snapshot model, not a reimplementation of Git's object format."""
import hashlib
import json
from dataclasses import dataclass, field

@dataclass
class Snapshot:
    work: dict = field(default_factory=dict)
    index: dict = field(default_factory=dict)
    commits: list = field(default_factory=list)

    def add(self, name):
        self.index[name] = self.work[name]

    def commit(self):
        content = dict(self.index)
        digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()[:10]
        self.commits.append((digest, content))
        return digest

if __name__ == "__main__":
    repo = Snapshot()
    repo.work["notes.txt"] = "v1"
    repo.add("notes.txt")
    repo.work["notes.txt"] = "v2"
    repo.commit()
    print("committed:", repo.commits[-1][1])
    print("working:", repo.work)
