"""Exercise the installed wheel outside the checkout, without network or Hub downloads."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_wheel_resources_and_inference_outside_the_checkout(tmp_path):
    root = Path(__file__).resolve().parents[1]
    wheels = tmp_path / "wheels"
    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            str(root),
            "--no-deps",
            "--no-build-isolation",
            "--no-index",
            "--wheel-dir",
            str(wheels),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    installed = tmp_path / "installed"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--target",
            str(installed),
            str(next(wheels.glob("*.whl"))),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # Isolated Python ignores PYTHONPATH and never adds the repository to sys.path.
    program = r"""
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from naltra.utils.resources import resource_path
from naltra.data.manifest import get_taxonomy_checksums, get_source_config
from naltra.models.naive_bayes import NaiveBayesModel
from naltra.models.svm import SVMModel
from naltra.models.bilstm import BiLSTMModel
from naltra.models.transformer import TransformerModel
from naltra.models.hybrid_knn import HybridKNNModel
from naltra.models.kev import KevModel
from naltra.models.jev import JevModel
from naltra.models.laya import LayaModel
from naltra.pipeline.prediction import PredictionPipeline
source = Path(sys.argv[2])
for name in ['taxonomy/taxonomy.json', 'taxonomy/label_map.json', 'configs/data.yaml',
             'configs/models/svm.yaml']:
    assert resource_path(name).read_bytes() == (source / name).read_bytes()
assert get_taxonomy_checksums()['taxonomy_version'] == '0.4.0'
assert get_source_config('translation')['revision']
for family in [SVMModel, BiLSTMModel, TransformerModel, HybridKNNModel,
               KevModel, JevModel, LayaModel]:
    family()
model = NaiveBayesModel({'n_jobs': 1, 'features': {'min_df': 1}})
rows = [{'id': str(i), 'text': text, 'labels': [label], 'labels_direct': [label], 'language': 'en',
         'source': 'fixture', 'license': 'MIT', 'split': 'train'}
        for i, (text, label) in enumerate([
            ('laser optics photon light', 'optics'),
            ('optical lens laser light', 'optics'),
            ('sound acoustic waves noise', 'acoustics'),
            ('acoustics sound wave sensor', 'acoustics')])]
model.train(rows)
model.save('artifact')
loaded = NaiveBayesModel()
loaded.load('artifact')
result = PredictionPipeline(model=loaded).predict('laser light photon')
assert set(result.label_scores) == {'acoustics', 'optics'}
assert result.hierarchy_paths
"""
    smoke = subprocess.run(
        [sys.executable, "-I", "-c", program, str(installed), str(root)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert smoke.returncode == 0, smoke.stdout + smoke.stderr
