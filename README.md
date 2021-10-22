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

for type in sentences text; do for file in */*/${type}.gz ; do filename=${file%.*} ; zcat $file | base64 -d > $filename; done ; done

Meta information is gzipped as usual:

for type in mime url; do for file in */*/${type}.gz ; do gzip -d $file ; done ; done

