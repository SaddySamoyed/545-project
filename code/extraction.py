# @title Extraction

nlp = spacy.load("en_core_web_sm")

def extract_nouns(sentence, top_n=5):
    doc = nlp(sentence)

    # 合并连续的名词为词组（如 "plant species"）
    phrases = []
    current_phrase = []
    for token in doc:
        if token.pos_ in ['NOUN', 'PROPN']:
            current_phrase.append(token.text)
        else:
            if current_phrase:
                phrases.append(" ".join(current_phrase))
                current_phrase = []
    # 处理最后可能残留的名词
    if current_phrase:
        phrases.append(" ".join(current_phrase))

    # 统计词频
    counts = Counter(phrases)
    return [phrase for phrase, _ in counts.most_common(top_n)]
sentence = "A policeman stops on a street with a search dog."
list_A = extract_nouns(sentence)
print("Extracted nouns/phrases:", list_A)