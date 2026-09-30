import inspect

from naltra.models.base import BaseNALTRAModel
from naltra.models.bilstm import BiLSTMModel
from naltra.models.jev import JevModel
from naltra.models.laya import LayaModel
from naltra.models.naive_bayes import NaiveBayesModel
from naltra.models.svm import SVMModel
from naltra.models.transformer import TransformerModel


def test_base_model_is_abstract() -> None:
    assert inspect.isabstract(BaseNALTRAModel)
    assert set(BaseNALTRAModel.__abstractmethods__) == {
        "train",
        "predict",
        "predict_batch",
        "save",
        "load",
    }


def test_all_model_placeholders_follow_the_common_interface() -> None:
    model_types = [
        NaiveBayesModel,
        SVMModel,
        BiLSTMModel,
        TransformerModel,
        JevModel,
        LayaModel,
    ]
    assert all(issubclass(model_type, BaseNALTRAModel) for model_type in model_types)
