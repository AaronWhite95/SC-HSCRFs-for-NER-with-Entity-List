from typing import Dict, List, Sequence, Iterable, Tuple
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.common.file_utils import cached_path
import logging
from overrides import overrides
import itertools
from allennlp.data.tokenizers import Token
from allennlp.data.fields import ListField, TextField, SequenceLabelField, Field, MetadataField, SpanField, LabelField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)
polar_dict = {
    "1": "Ture",
    "0": "False"
}

@DatasetReader.register("bertclassification")
class ClassificationReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer] = None,
        lazy: bool = False
    ) -> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
    
    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
        file_path = cached_path(file_path)

        with open(file_path, "r") as data_file:
            logger.info("Reading instances from lines in file at: %s", file_path)
            for line in data_file:
                polar, sent = line.strip().split(",")
                tokens = [Token(token) for token in sent]
                yield self.text_to_instance(tokens, polar)
    
    def text_to_instance(
        self,
        tokens:List[Token],
        polar
    ) -> Instance:
        sequence = TextField(tokens, self._token_indexers)
        instance_fields: Dict[str, Field] = {'tokens': sequence}
        instance_fields['label'] = LabelField(polar)
        return Instance(instance_fields)             
