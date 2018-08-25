from typing import Dict, Optional
import logging

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register("sentence_classifier")
class AcademicSentenceClassifier(Model):
    """
    This ``Model`` performs text classification for a sentence from an academic paper.  We assume we're given a
    title and a sentence, and we predict some output label.

    The basic model structure: we'll embed the title and the sentence, and encode each of them with
    separate Seq2VecEncoders, getting a single vector representing the content of each.  We'll then
    concatenate those two vectors, and pass the result through a feedforward network, the output of
    which we'll use as our scores for each label.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 title_encoder: Seq2VecEncoder,
                 sentence_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(AcademicSentenceClassifier, self).__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.num_classes = self.vocab.get_vocab_size("labels")
        #self.num_classes = 7 
        self.title_encoder = title_encoder
        self.sentence_encoder = sentence_encoder
        self.classifier_feedforward = classifier_feedforward

        logger.info("num_classes: %s", str(self.num_classes))

        if text_field_embedder.get_output_dim() != title_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the title_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            title_encoder.get_input_dim()))
        if text_field_embedder.get_output_dim() != sentence_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the text_field_embedder must match the "
                                     "input dimension of the abstract_encoder. Found {} and {}, "
                                     "respectively.".format(text_field_embedder.get_output_dim(),
                                                            abstract_encoder.get_input_dim()))
        self.metrics = {
                "accuracy": CategoricalAccuracy(),
                "accuracy3": CategoricalAccuracy(top_k=3)
        }
        self.loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                title: Dict[str, torch.LongTensor],
                sentence: Dict[str, torch.LongTensor],
                label: torch.LongTensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        title : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        sentence : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.

        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_title = self.text_field_embedder(title)
        title_mask = util.get_text_field_mask(title)
        encoded_title = self.title_encoder(embedded_title, title_mask)

        embedded_sentence = self.text_field_embedder(sentence)
        sentence_mask = util.get_text_field_mask(sentence)
        encoded_sentence = self.sentence_encoder(embedded_sentence, sentence_mask)

        logits = self.classifier_feedforward(torch.cat([encoded_title, encoded_sentence], dim=-1))
        output_dict = {'logits': logits}
        if label is not None:
            loss = self.loss(logits, label)
            for metric in self.metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()}

    @classmethod
    def from_params(cls, vocab: Vocabulary, params: Params) -> 'AcademicSentenceClassifier':
        embedder_params = params.pop("text_field_embedder")
        text_field_embedder = TextFieldEmbedder.from_params(vocab, embedder_params)
        title_encoder = Seq2VecEncoder.from_params(params.pop("title_encoder"))
        sentence_encoder = Seq2VecEncoder.from_params(params.pop("sentence_encoder"))
        classifier_feedforward = FeedForward.from_params(params.pop("classifier_feedforward"))

        initializer = InitializerApplicator.from_params(params.pop('initializer', []))
        regularizer = RegularizerApplicator.from_params(params.pop('regularizer', []))

        return cls(vocab=vocab,
                   text_field_embedder=text_field_embedder,
                   title_encoder=title_encoder,
                   sentence_encoder=sentence_encoder,
                   classifier_feedforward=classifier_feedforward,
                   initializer=initializer,
                   regularizer=regularizer)
