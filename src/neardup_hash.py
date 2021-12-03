from unicodedata import category as cat
from unidecode import unidecode
from xxhash import xxh64
import sys

# Translate table to remove non alphabetic characters
tbl = [chr(i) for i in range(sys.maxunicode) if not cat(chr(i)).startswith('L')]
remove_non_alpha = str.maketrans('', '', ''.join(tbl))

# Remove near duplicates, only print ones that were not duplicates
hashes = {}
for line in sys.stdin:
    tmp = xxh64(unidecode(line.strip("\n").lower().translate(remove_non_alpha))).hexdigest()
    hsh = tmp.split('\t')[-1]
    if hsh not in hashes:
        print(line.strip())
        hashes[hsh] = 1
