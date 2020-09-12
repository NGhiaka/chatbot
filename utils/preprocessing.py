import re

def preprocess_senctence(senttence):
    sentence = senttence.lower().strip()
    # senttence = sentence.replace(' u ', ' you ')
    # create space between a word and the punctuation following it
    # eg: "this is a cat." => "this is a cat ." 
    sentence = re.sub(r"([?.!,])", r" \1", sentence)
    # replace everything
    sentence = re.sub(r"([^A-Za-z?.!,$])", " ", sentence)
    sentence = sentence.strip()
    # print(sentence)
    return sentence