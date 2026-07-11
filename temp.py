import os
from pathlib import Path


class ProjectScanner:
    def __init__(
        self,
        root_dir=".",
        batch_size=10,
        max_length=255,
        relative_path=True,
    ):
        self.root_dir = Path(root_dir).resolve()
        self.batch_size = batch_size
        self.max_length = max_length
        self.relative_path = relative_path

        self.items = []
        self.index = 0

        self._scan()

    def _scan(self):
        self.items.clear()

        for root, dirs, files in os.walk(self.root_dir):
            for d in dirs:
                path = Path(root) / d
                self.items.append(self._format(path))

            for f in files:
                path = Path(root) / f
                self.items.append(self._format(path))

    def _format(self, path):
        if self.relative_path:
            text = str(path.relative_to(self.root_dir))
        else:
            text = str(path)

        if len(text) > self.max_length:
            text = text[: self.max_length - 3] + "..."

        return text

    def next_batch(self):
        if self.index >= len(self.items):
            return []

        batch = self.items[self.index : self.index + self.batch_size]
        self.index += self.batch_size
        return batch

    def prev_batch(self):
        self.index = max(0, self.index - self.batch_size * 2)
        return self.next_batch()

    def reset(self):
        self.index = 0

    def total_items(self):
        return len(self.items)

scanner = ProjectScanner("D:\project\RE\RE")
print(scanner.items)