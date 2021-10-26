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
for type in sentences text; do for file in */*/${type}.gz ; do filename=${file%.*} ; zcat $file | base64 -d > $filename; done ; done
```

Meta information is gzipped as usual:

```
for type in mime url; do for file in */*/${type}.gz ; do gzip -d $file ; done ; done
```


## Cleaning

Clean all the data:

```
for file in data/crawl_19_10_2021/*/en/sentences; do echo $file ; python src/clean_data.py --input_file $file -ml 7 ;  done
```

## Training and evaluating LMs

Install [KenLM](https://github.com/kpu/kenlm/) and its dependencies.

Train a LM, with trigrams:

```
./bin/lmplz -o 3 -S 100G -T /tmp < ../../../data/crawl_19_10_2021/cybersecurity/en/sentences.clean > ../../../data/crawl_19_10_2021/cybersecurity/en/sentences.arpa
```

Make the model a binary:

```
./bin/build_binary ../../../data/crawl_19_10_2021/cybersecurity/en/sentences.arpa ../../../data/crawl_19_10_2021/cybersecurity/en/sentences.binary
```

Apply on (same) data to get perplexities:

```
./bin/query ../../../data/crawl_19_10_2021/cybersecurity/en/sentences.binary < ../../../data/crawl_19_10_2021/cybersecurity/en/sentences.clean
```

Run everything at once, including evaluation on all other DSIs:

```
./src/run_kenlm.sh
```

Print all the perplexities in a nice table:

```
python src/perplexity_table.py --input_folder data/crawl_19_10_2021/
```
