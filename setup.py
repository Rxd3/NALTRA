"""Copy canonical runtime resources into wheels without duplicating source files."""

from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPy(build_py):
    def run(self):
        super().run()
        root = Path(__file__).resolve().parent
        target = Path(self.build_lib) / "naltra" / "resources"
        for directory, pattern in (("configs", "*.yaml"), ("taxonomy", "*.json")):
            for source in (root / directory).rglob(pattern):
                destination = target / source.relative_to(root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                self.copy_file(str(source), str(destination))


setup(cmdclass={"build_py": BuildPy})
