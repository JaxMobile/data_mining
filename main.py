import py_stringmatching as sm


test_string = ' .hel32lo, world!! data, science, is amazing!!. hello.'

alphabet_tok = sm.AlphabeticTokenizer()
print(alphabet_tok.tokenize(test_string))

delim_tok = sm.DelimiterTokenizer(delim_set=[','])
print(delim_tok.tokenize(test_string))

alnum_tok = sm.AlphanumericTokenizer()
print(alnum_tok.tokenize(test_string))