# Explaining predictions with LIME

In this repository we train a model to classify sentences from academic papers using [AllenNLP](https://github.com/allenai/allennlp) and explain the predictions using [LIME](https://github.com/marcotcr/lime). 

To train the model, run this:
```
python3 run.py train experiments/sentence_classifier_architecture.json  --include-package sentence_classifier.dataset_readers --include-package sentence_classifier.models -s ./trained_model
```

And this is how we evaluate the model:
```
python3 run.py evaluate trained_model/model.tar.gz  --include-package sentence_classifier.dataset_readers --include-package sentence_classifier.models --evaluation-data-file data/test/
```

The notebook has the code to explain predictions with LIME.

![prediction result and explanation with LIME]("example.png")


