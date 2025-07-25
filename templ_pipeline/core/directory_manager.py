
import tempfile
import shutil
from pathlib import Path

class DirectoryManager:
    def __init__(self, base_name, auto_cleanup=True, centralized_output=True, output_root=None):
        self.base_name = base_name
        self.auto_cleanup = auto_cleanup
        self.centralized_output = centralized_output
        self.output_root = Path(output_root) if output_root else Path(tempfile.gettempdir())
        self.directory = self.output_root / self.base_name
        self.directory.mkdir(parents=True, exist_ok=True)

    def cleanup(self):
        if self.auto_cleanup:
            shutil.rmtree(self.directory)
