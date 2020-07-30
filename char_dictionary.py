'''
CHARACTER LEVEL DICTIONARY
'''

id2char = '`~!@#$%^&*()_+1234567890-=qwertyuiop[]|";:/?.>,<}{asdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM' + " '"

# 0 = pad
# 1 = unknown 0 is pad (+offset from char dictionary)
char2id = { w: i + 2 for i, w in enumerate(id2char) }

vocab_size = len(id2char) + 2

def encode_char(char): 
    return char2id.get(char, 1)

def decode_token(token):
    if token == 0 or token == 1:
        return '<PAD>' if token == 0 else '<UNK>'
    return id2char[token-2]

def encode_string(string):
    return [encode_char(c) for c in string]

def decode_tokens(tokens):
    return ''.join([decode_token(t) for t in tokens])
