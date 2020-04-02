from spacy.lang.en import English

tokenizer = English()
tokenizer.add_pipe(tokenizer.create_pipe("sentencizer"))

def clean_text(text):
    text = text.replace("]", " ] ")
    text = text.replace("[", " [ ")
    text = text.replace("\n", " ")
    text = text.replace("''", '" ').replace("``", '" ')

    return text

def word_tokenize(text):
    tokens = [token.text for token in tokenizer(text) if token.text]
    tokens = [t for t in tokens if t.strip("\n").strip()]
    return tokens


def sent_tokenize(text):
    return [[token.text for token in sentence if token.text] for sentence in tokenizer(text).sents]


def feature_tokenize(text, f_sep=u"￨"):
    return [t.split(f_sep)[0] for t in text.split()], [t.split(f_sep)[1] for t in text.split()]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)

    return spans