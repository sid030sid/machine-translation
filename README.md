# NLP: English-Dutch Machine Translator

## About
This project creates several models for translating English to Dutch and vis-a-vis. After creation, all models are tested and compared to each other - especially how they respectively perform depending on the the different word embeding techniques. This project's models are:
1. a word based neural network (for English to Dutch and for Dutch to English)
2. a word based neural network with attention
3. a character based machine translation model


## Requirements and installations
1. install Phyton 3
2. download English-Dutch corpus from [statmt.org](https://www.statmt.org/europarl/v7/nl-en.tgz)
3. add folder with name data and insert the previously downloaded data: ```europarl-v7.nl-en.en``` and ```europarl-v7.nl-en.nl``` into it


## Get started
1. create virtual environment by entering in terminal: py -3 -m venv .venv
2. activate newly created environment by entering in terminal: .venv\scripts\activate
3. download packages by entering in terminal with active virtual environment: python -m pip install pandas, numpy, keras, tensorflow dataframe_image, gensim, keras-self-attention, seaborn
4. run phyton file by doing right click on respective file and selecting "Run Phyton file in terminal" or alternatively by entering in terminal witch active virtual environment: python file_name.p


## Data
The data for this project is based on “European Parliament Proceedings Parallel Corpus 1996-2011”.
The Europarl parallel corpus is extracted from the proceedings of the European Parliament. It includes
versions in 21 European languages: Romanic (French, Italian, Spanish, Portuguese, Romanian),
Germanic (English, Dutch, German, Danish, Swedish), Slavik (Bulgarian, Czech, Polish, Slovak, Slovene),
Finni-Ugric (Finnish, Hungarian, Estonian), Baltic (Latvian, Lithuanian), and Greek.
All models in this project are based on the parallel data containing English and Dutch sentences from https://www.statmt.org/europarl/v7/nl-en.tgz. You can find more details about the data in https://www.statmt.org/europarl/. Please note that only 10% of the data is used to train and test all models since the dataset is too large. Said 10% is randomly selected after import.
