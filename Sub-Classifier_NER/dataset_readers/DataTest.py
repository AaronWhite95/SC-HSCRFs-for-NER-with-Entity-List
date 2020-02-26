from allennlp.common.testing import AllenNlpTestCase
from span_conll2003_dataset_reader import SpanConll2003DatasetReader

reader = SpanConll2003DatasetReader()
dataset = reader.read("../data/train.txt")
print(dataset[0])