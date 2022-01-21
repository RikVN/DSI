#!/bin/bash
set -eu -o pipefail
# Download the 2 models
wget "https://www.let.rug.nl/rikvannoord/DSI/en_deberta.zip"
wget "https://www.let.rug.nl/rikvannoord/DSI/ml_xlmr.zip"
# Create correct directories
mkdir -p models
mkdir -p models/en_model models/ml_model 
# Unpack models in correct folder and clean up zip files, no longer needed
mv en_deberta.zip models/en_model/
mv ml_xlmr.zip models/ml_model/
cd models/en_model ; unzip en_deberta.zip ; rm en_deberta.zip ; cd ../../
cd models/ml_model ; unzip ml_xlmr.zip ; rm ml_xlmr.zip ; cd ../../
