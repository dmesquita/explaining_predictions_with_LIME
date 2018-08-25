from typing import Dict
import logging
import xml.etree.ElementTree as ET

from overrides import overrides

import tqdm
import os
import re

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("az_papers")
class AZDatasetReader(DatasetReader):
    """
    Reads a xml files containing papers and creates the dataset.

    """
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, folder_path):
        files = [folder_path+f for f in os.listdir(folder_path) if ".az-scixml" in f]
        logger.info("Reading instances from lines in file at: %s", folder_path)
        for f_ in files:
            with open(f_, "r") as f:
                raw_text = f.read()
                temp_text1 = re.sub('<REF.*?</REF>', '', raw_text) 
                temp_text2 = re.sub('<CREF/>', '', temp_text1) 
                clean_text = re.sub('<EQN/>', '', temp_text2)
                
                tree = ET.fromstring(clean_text)
                
                for sentence in tree.findall(".//S"):
                    temp = {}
                    
                    temp["year"] = tree.find(".//YEAR").text.strip()
                    temp["fileno"] = tree.find(".//FILENO").text.strip()
                    title = tree.find(".//TITLE").text.strip()
                    try:
                        label = sentence.attrib["AZ"]
                    except KeyError as error:
                        print('AZ label not found on sentence: "%s", file "%s"' % (sentence.text,f_))
                        continue
                    
                    sent = sentence.text.strip()
                    temp["source"] = "body"
                    
                    yield self.text_to_instance(title, sent, label)
                    
                for sentence in tree.findall(".//A-S"):
                    temp = {}
                    
                    temp["year"] = tree.find(".//YEAR").text.strip()
                    temp["fileno"] = tree.find(".//FILENO").text.strip()
                    title = tree.find(".//TITLE").text.strip()
                    label = sentence.attrib["AZ"]
                    sent = sentence.text.strip()
                    temp["source"] = "abstract"
                    
                    yield self.text_to_instance(title, sent, label)

    @overrides
    def text_to_instance(self, title: str, sentence: str, label: str = None) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        tokenized_title = self._tokenizer.tokenize(title)
        tokenized_sentence = self._tokenizer.tokenize(sentence)
        title_field = TextField(tokenized_title, self._token_indexers)
        sentence_field = TextField(tokenized_sentence, self._token_indexers)
        fields = {'title': title_field, 'sentence': sentence_field}
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)

    @classmethod
    def from_params(cls, params: Params) -> 'AZDatasetReader':
        lazy = params.pop('lazy', False)
        tokenizer = Tokenizer.from_params(params.pop('tokenizer', {}))
        token_indexers = TokenIndexer.dict_from_params(params.pop('token_indexers', {}))
        params.assert_empty(cls.__name__)
        return cls(lazy=lazy, tokenizer=tokenizer, token_indexers=token_indexers)
