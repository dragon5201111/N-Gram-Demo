import re
import json

from collections import Counter


class CorpusPreprocessor:
    @classmethod
    def preprocess_corpus(cls, corpus : str) -> str:
        corpus = re.sub(r"[^a-zA-Z0-9 ]+", "", corpus)
        corpus = re.sub(r"\s+", " ", corpus)
        return corpus.strip().lower()


class NGramTokenizer:
    @classmethod
    def generate(cls, corpus : str, n : int) -> list[tuple[str,...]]:
        if n <= 0:
            raise ValueError("n must be a positive integer")

        tokens = corpus.split()
        return_tokens = []

        num_tokens = len(tokens) - (n - 1)
        for i in range(num_tokens):
            return_tokens.append(tuple(tokens[i: i + n]))

        return return_tokens

class NGramModel:
    def __init__(self, corpus : str, vocabulary : set[str], n : int):
        if n <= 0:
            raise ValueError("n must be a positive integer")
        self.n = n
        self.corpus = corpus
        self.vocabulary = vocabulary
        self.n_gram_frequencies = Counter(NGramTokenizer.generate(corpus, n))
        self.lower_order_frequencies = Counter(NGramTokenizer.generate(corpus, n - 1))

    def get_probability_a_given_b(self, a: tuple[str, ...], b: tuple[str, ...]) -> float:
        b_and_a_frequency = self.n_gram_frequencies.get(b + a, 0)
        b_frequency = self.lower_order_frequencies.get(b, 0)

        if b_frequency == 0:
            return 0.0

        return b_and_a_frequency / b_frequency

    def predict_next_word(self, context: tuple[str, ...], default="<unk>") -> str:
        max_probability = 0.0
        best_word = None

        for word in self.vocabulary:
            a = (word,)
            probability = self.get_probability_a_given_b(a, context)

            if probability > max_probability:
                max_probability = probability
                best_word = word

        return best_word if best_word is not None else default

    def generate_words(self, context: tuple[str, ...], count=1) -> list[str]:
        generated_words = list(context)
        slide = self.n - 1

        for i in range(count):
            next_word = self.predict_next_word(context)
            generated_words.append(next_word)
            context = tuple(generated_words[-slide:])

        return generated_words


def t_swift_lyrics_json_to_str():
    result = ""
    with open("album-song-lyrics.json", "r", encoding="utf-8") as t_swift_lyrics:
        t_swift_lyrics_json = json.load(t_swift_lyrics)

        for album in t_swift_lyrics_json:
            for song in album["Songs"]:
                for lyric in song["Lyrics"]:
                    result = f"{result} {lyric['Text']}"
    return result

if __name__ == "__main__":
    seed_text = "And you come away with a   "
    context = tuple(word.lower() for word in seed_text.strip().split())
    n = len(context) + 1

    t_swift_corpus = CorpusPreprocessor.preprocess_corpus(t_swift_lyrics_json_to_str())
    n_gram_model = NGramModel(n=n, corpus=t_swift_corpus, vocabulary=set(t_swift_corpus.split()))

    words_to_generate = 15
    words_generated = " ".join(n_gram_model.generate_words(context, count=words_to_generate))
    print(words_generated)







