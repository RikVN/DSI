'''Things we import in the scripts, but might want to change'''
punctuation = ['.', '?', '!', ':', ';']
replacements = [["‘", "'"], ["’", "'"], ['“', '"'], ['”', '"'], ['`', "'"], ["„", '"'],
               ["''", '"'], ['–', '-'], ['—', '-'], ['\t', ' ']]
# Download from here: https://github.com/dwyl/english-words
word_list_file = "data/english_words/words_dictionary.json"
dsis = ["cybersecurity", "e-health", "eessi", "online-dispute-resolution", "safer-internet", "e-justice", "europeana", "open-data-portal", "other"]
small_dsis = ["cyber", "health", "eessi", "odr", "safe", "e-just", "europ", "open", "other"]
to_small = {}
for sm, ds in zip(small_dsis, dsis):
    to_small[ds] = sm
