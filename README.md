# DSI classification

Code for the DSI experiments in the MaCoCu project. Corresponding paper will soon be available.

First, create a Conda environment for this project:

```
conda create -n dsi python=3.7
conda activate dsi
```

And install the requirements:

```
pip install -r requirements.txt
```

## Data

If you are only interested in doing DSI classification, you can just download the data:

```
wget "https://www.let.rug.nl/rikvannoord/DSI/v1.zip"
unzip v1.zip
```

If you want to preprocess your own set of sentences (in $FILE) exactly the way did, do the following:

```
./src/preprocess.sh $FILE
```

However: if you are only interested in using our trained models, you do not want to do this.

This does all sorts of filtering that you probably do not want, such as removing sentences smaller and larger than a certain length.

In that case, we still recommend doing the normalization step (as that is what the model expects). Run the following:

```
python src/clean_data.py -i $FILE -o ${FILE}.clean --only_normalize
```

## Using a trained model

First, download our pretrained models and put them in a ``models`` folder:

```
./src/setup_models.sh
```

Our best English model is based on DeBERTa-v3, while the multi-lingual model is based on XLM-R. Note that we evaluated the latter model on Spanish and Dutch, but it will work on all languages in XLM-R.

Then simply run the parsing script by specifying the language, the model and the sentence file, say for English:

```
python src/lm_parse.py -l en -m models/en_model/ -s ${FILE}.clean
```

And for Spanish or Dutch (or any other language in XLM-R):

```
python src/lm_parse.py -l es -m models/ml_model/ -s ${FILE}.clean
```

You can find the final predictions in ``${FILE}.clean.pred``, and the softmax probabilities in ``${FILE}.clean.pred.prob``.

### Selecting predictions

Now that you have the predicted probabilities, you can select your own threshold for including sentences in your data set. You can use ``src/select_preds.py`` for this.

For selecting all sentences with a probability of >= 0.5 for all DSIs:

```
mkdir -p out
python src/select_preds.py -p ${FILE}.clean.pred.prob -s ${FILE}.clean -o out/ -min 0.5
```

The ``out/`` folder now contains the files per DSI, .txt for just the texts and .info for tab-separated file with all info as well.

If we are only interested in e-health and e-justice, we can do this:

```
python src/select_preds.py -p ${FILE}.clean.pred.prob -s ${FILE}.clean -o out/ -min 0.5 -d e-health e-justice
```

If you just want to get a sense of how many docs you would get for each DSI, you can print a stats table like this:

```
python src/select_preds.py -p ${FILE}.clean.pred.prob
```

## Training your own finetuned LM

For training your own model you have to specify configuration files. They work together with the ``configs/default.sh file``, containing all default settings. In your own configuration file, you can override certain settings by simply including them there.

For example, check out ``configs/en_test.sh``. This trains a bert-base model on the English data, and evaluates on the English dev set. Since it's just a test, we specify that we downsample each category to 200 instances, and only train for 1 epoch. You can run it like this:

```
mkdir -p exps
./src/train_lm.sh configs/en_test.sh
```

You can find all experimental files in ``exps/en_test/``, including log files, output files, the trained models and evaluation files.

For Spanish and Dutch, an example is added for doing zero-shot classification with a multi-lingual LM: ``configs/ml_test.sh``.

First, we create a dev set that is a combination of both the Spanish and Dutch sets:

```
mkdir -p v1/es-nl/
cat v1/es/dev v1/nl/dev > v1/es-nl/dev
shuf v1/es-nl/dev > v1/es-nl/dev.shuf
```

Then we simply train the model with the configuration file again, and evaluate on both the Spanish and Dutch dev sets individually:

```
./src/train_lm.sh configs/ml_test.sh
```

If you want to train the exact same models we did, use the config files ``configs/en_best.sh`` or ``configs/ml_best.sh``. You will have to do this on GPU and it will take 2 to 3 days.

Our train script automatically evaluates on the specified dev/test sets, but you can also run this separately, and plot a confusion matrix:

```
python src/eval.py -g $GOLD_FILE -p $PRED_FILE -c cm.png
```

If you want to evaluate on Spanish or Dutch, you'd likely want to fix how the eval script calculates the macro-average, as not all categories are present in the dev/test sets. Save the classification report first, and run this:

```
python src/fix_clf_report.py -i clf.txt
```

## Baseline models

You can also run a basic classifier by running the following (downsample to speed up):

```
python src/basic_classifiers.py -i v1/en/train -t v1/en/dev -tf -d 3000
```

This train a LinearSVM with unigrams and bigrams using a af TF-IDF vectorizer.

Note that there are a couple of important command line options to set, you can check them out by adding -h.

For example, if you want to see the best features:

```
python src/basic_classifiers.py -i v1/en/train -t v1/en/dev -tf -d 3000 --features
```
