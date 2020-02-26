from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor

from allennlp.data.tokenizers import Token
from allennlp.models.archival import load_archive

@Predictor.register("CON_NER_predictor")
class ConNERPredictor(Predictor):        
    def predict_json(self, inputs: JsonDict) -> [JsonDict]:
        '''
        sent = "2019年11月20日至22日，将在意大利罗马举行第四届系统可靠性和安全国际会议(ICSRS 2019)。过去3年内，ICSRS在巴黎、米兰、巴塞罗那等大城市举办。"
        tokens = [Token(token) for token in sent]
        tags = ["O" * len(tokens)]
        inputs = {
            "tokens": token,
            "tags": tags
        }
        '''
        #instance = self._dataset_reader.text_to_instance(tokens, tags)
        instances = self._batch_json_to_instances(inputs)
        #output_dict = self.predict_instance(instance)
        output_dict = self.predict_batch_instance(instances)
        print(output_dict), exit(0)
        return output_dict
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sent = json_dict["sent"]
        tokens = [Token(token) for token in sent]
        tags = ["O"] * len(tokens)
        #print(self._dataset_reader.text_to_instance(tokens, tags))
        return self._dataset_reader.text_to_instance(tokens, tags)



class Demo():
    def predict_result(self):
        archive = load_archive('../pure_hscrf/model.tar.gz')
        predictor = Predictor.from_archive(archive, 'CON_NER_predictor')
        exit(0)
        while True:
            sent = input()
            if sent == "":
                exit(0)
            tokens = [Token(token) for token in sent]
            tags = ["O" * len(tokens)]
            instance = reader.text_to_instance(tokens, tags)
            result = predictor.predict_json(instance)
            print(result)


if __name__ =="__main__":
    demo = Demo()
    demo.predict_result()
