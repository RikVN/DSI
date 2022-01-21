'''Things we import in the scripts, but might want to change'''
# What we consider punctuation when cleaning
punctuation = ['.', '?', '!', ':', ';']
# Replacement cleaning we perform
replacements = [["‘", "'"], ["’", "'"], ['“', '"'], ['”', '"'], ['`', "'"], ["„", '"'],
               ["''", '"'], ['–', '-'], ['—', '-'], ['\t', ' '], ["•", "."], ["●", "."]]
# Download from here: https://github.com/dwyl/english-words
word_list_file = "data/english_words/words_dictionary.json"
dsis = ["cybersecurity", "e-health", "eessi", "online-dispute-resolution", "safer-internet", "e-justice", "europeana", "open-data-portal", "other"]
small_dsis = ["Cyber", "Health", "EESSI", "ODR", "Safe", "E-just", "Europ", "Open", "Other"]
to_small = {}
for sm, ds in zip(small_dsis, dsis):
    to_small[ds] = sm
