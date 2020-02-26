from allennlp.common import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.models import Model
from allennlp.predictors import Predictor
from overrides import overrides
import numpy as np
from allennlp.data.tokenizers import Token
from allennlp.models.archival import load_archive

# You need to name your predictor and register so that `allennlp` command can recognize it
# Note that you need to use "@Predictor.register", not "@Model.register"!
@Predictor.register("sentence_classifier_predictor")
class SentenceClassifierPredictor(Predictor):
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self.model = model

    def predict_json(self, inputs: JsonDict) -> [JsonDict]:
        
        instances = self._batch_json_to_instances(inputs)
        output_dict = self.predict_batch_instance(instances)
        label_id = [np.argmax(dic["logits"]) for dic in output_dict]
        id2label = self.model.vocab.get_index_to_token_vocabulary('labels')
        return [id2label[id] for id in label_id]

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sent = json_dict["sent"]
        tokens = [Token(token) for token in sent]
        polar = "0"
        return self._dataset_reader.text_to_instance(tokens, polar)
