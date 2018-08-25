from typing import Tuple

from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.service.predictors.predictor import Predictor

@Predictor.register('sentence-classifier')
class AcademicSentenceClassifierPredictor(Predictor):
    """"Predictor wrapper for the AcademicSentenceClassifier"""
    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        title = json_dict['title']
        sentence = json_dict['sentence']
        instance = self._dataset_reader.text_to_instance(title=title, sentence=sentence)

        # label_dict will be like {0: "ACL", 1: "AI", ...}
        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')
        label_dict =  {0:"OWN", 1:"OTH",2: "BKG",3: "CTR",4:  "AIM", 5: "TXT", 6: "BAS"}
        # Convert it to list ["ACL", "AI", ...]
        all_labels = [label_dict[i] for i in range(len(label_dict))]

        return instance, {"class_names": label_dict}
