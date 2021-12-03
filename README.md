# DSI

Code for the DSI experiments in the MaCoCu project

Create a Conda environment for this project:

```
conda create -n macocu python=3.7
```

And install the requirements:

```
pip install -r requirements.txt
```

## Data handling

Unpack the data files, text is in base64:

```
for type in clean_sentences clean_text; do for file in */*/${type}.gz ; do filename=${file%.*} ; zcat $file | base64 -d > $filename; done ; done
```

Meta information is gzipped as usual:

```
for type in clean_mime clean_url; do for file in */*/${type}.gz ; do gzip -d $file ; done ; done
```

## Cleaning and splitting

Make next commands a bit easier and customizable:

```
export fol="data/dsi_clean"
```

If you want to include a non-DSI category it's best to already include a file called "sentences" in the folder "other" in the same structure. In our case, we used 150k sentences from the Dutch-English paracrawl.
Clean all the data:

```
for file in ${fol}/*/en/sentences; do echo $file ; python src/clean_data.py --input_file $file -minl 7 -maxl 50 ;  done
```

Tokenizing and lowercasing:

```
for file in ${fol}/*/en/sentences.clean; do echo $file ; python src/tokenize_and_lowercase.py -i ${file} -o ${file}.tok.lower -l en ; done
```

Run deduplication on the data:

```
for file in ${fol}/*/en/sentences.clean.tok.lower; do echo $file ; cat $file | python src/neardup_hash.py > ${file}.dedup ; done
```

Split in train/test sets for LM training:

```
python src/split_data.py --input_folder $fol/ -d 5000 -t 5000 -fa _lm
```

But also in train/dev/test for classifier training:

```
python src/split_data.py --input_folder $fol/ -d 1000 -t 1000 -fa _clf -al -mo
```

Now we combine the classification splits in single files and shuffle them:

```
mkdir ${fol}/all/
for type in train dev test; do cat ${fol}/*/en/${type}_clf*dedup > ${fol}/all/${type}.clf ; done
for type in train dev test; do shuf ${fol}/all/${type}.clf > ${fol}/all/${type}.clf.shuf ; done
```

## DSI classification

I assume you created the data splits for DSI classification. To run a basic classifier, run the following:

```
python src/basic_classifiers.py --input_file ${fol}/all/train.clf.shuf --test_file ${fol}/all/dev.clf.shuf
```

This train a LinearSVM with unigrams and bigrams, with each feature at least occurring 5 times.

Note that there are a couple of important command line options to set:

```
-tf, --tfidf		  Use the TF-IDF vectorizer instead of CountVectorizer
-a  ,--algorithm      Use "nb" for Naive Bayes and "svm" for LinearSVM
-cv, --cross_validate Specify number of folds for cross-validation instead of dev/test prediction
-f, --features        Print best features per class (only works for svm)
-d, --down_sample	  Downsample all non-other classes to max this amount
-l, --limit_train     Limit training set to this amount of instances (if applicable after downsampling)
-cm, --confusion	  Save plot of confusion matrix here, if not added do not plot
-ovr, --one_vs_rest   Do one vs rest classification instead of one vs one (default)
```

There are more options, you can check them out by adding -h.

It's also possible to fine-tune a pretrained language model. You should do this on a GPU. It can take quite a long time, so you potentially might want to use -l or -d here.

```
python src/lm_classifier.py --train_file ${fol}/all/train.clf.shuf --dev_file ${fol}/all/dev.clf.shuf -l 10000
```

This trains a model with bert-base-uncased (default), but the LM-string can be specified with -lm (as long as it's in AutoModelForSequenceClassification in the transformers library. You can use similar arguments as above for down-sampling and limiting the training set.

## Training and evaluating LMs

Install [KenLM](https://github.com/kpu/kenlm/) and its dependencies. I assume it's in the following subfolder of this repo:

```
cd tools/kenlm/build
```

Train a LM, with trigrams:

```
./bin/lmplz -o 3 -S 100G -T /tmp < ../../../data/dsi_clean/cybersecurity/en/sentences.clean > ../../../data/dsi_clean/cybersecurity/en/sentences.arpa
```

Make the model a binary:

```
./bin/build_binary ../../../data/dsi_clean/cybersecurity/en/sentences.arpa ../../../data/dsi_clean/cybersecurity/en/sentences.binary
```

Apply on (same) data to get perplexities:

```
./bin/query ../../../data/dsi_clean/cybersecurity/en/sentences.binary < ../../../data/dsi_clean/cybersecurity/en/sentences.clean
```

To make things easier we can run everything at once, including evaluation on all other DSIs:

```
./src/run_kenlm.sh exp/exp_name/
```

The given folder is created and contains all the important experimental files (check data/ log/ and eval/).

Print all the perplexities in a nice table (note run_kenlm.sh already does this):

```
python src/perplexity_table.py --input_folder exp/exp_name/data/
```
