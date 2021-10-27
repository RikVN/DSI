'''Simply functions that we re-use often'''

import sys
from spacy.lang.en import English
from spacy.lang.es import Spanish
from spacy.lang.nl import Dutch


def write_to_file(lst, out_file):
	'''Write list to file'''
	with open(out_file, "w") as out_f:
		for line in lst:
			out_f.write(line.strip() + '\n')
	out_f.close()


def setup_spacy_tokenizer(lang):
	'''Given a language code, return the correct tokenizer'''
	if lang == "en":
		nlp = English()
	elif lang == "es":
		nlp = Spanish()
	elif lang == "nl":
		nlp = Dutch()
	else:
		raise ValueError("Language {0} not supported so far".format(lang))
	return nlp.tokenizer
