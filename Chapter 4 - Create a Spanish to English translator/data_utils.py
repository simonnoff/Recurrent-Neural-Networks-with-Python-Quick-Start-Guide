# |------------------------------------------------|
# |Creates a dataset from raw text files           |
# |------------------------------------------------|
# |Dataset: OpenSubtitles                          |
# |Url: http://opus.lingfil.uu.se/OpenSubtitles.php|
# |Credits to: Nemanja Tomic (https://github.com/Nemzy/language-translation)
# |------------------------------------------------|

import pickle
from collections import Counter

def read_sentences(file_path):
	sentences = []

	with open(file_path, 'r') as reader:
		for s in reader:
			sentences.append(s.strip())

	return sentences

def create_dataset(english_sentences, spanish_sentences):

	english_vocab_dict = Counter(word.strip(',." ;:)(][?!') for sentence in english_sentences for word in sentence.split())
	spanish_vocab_dict = Counter(word.strip(',." ;:)(][?!') for sentence in spanish_sentences for word in sentence.split())

	english_vocab = list(map(lambda x: x[0], sorted(english_vocab_dict.items(), key = lambda x: -x[1])))
	spanish_vocab = list(map(lambda x: x[0], sorted(spanish_vocab_dict.items(), key = lambda x: -x[1])))

	english_vocab = english_vocab[:20000]
	spanish_vocab = spanish_vocab[:30000]

	start_idx = 2
	english_word2idx = dict([(word, idx+start_idx) for idx, word in enumerate(english_vocab)])
	english_word2idx['<ukn>'] = 0
	english_word2idx['<pad>'] = 1

	english_idx2word = dict([(idx, word) for word, idx in english_word2idx.items()])

	start_idx = 4
	spanish_word2idx = dict([(word, idx+start_idx) for idx, word in enumerate(spanish_vocab)])
	spanish_word2idx['<ukn>'] = 0
	spanish_word2idx['<go>']  = 1
	spanish_word2idx['<eos>'] = 2
	spanish_word2idx['<pad>'] = 3

	spanish_idx2word = dict([(idx, word) for word, idx in spanish_word2idx.items()])

	x = [[english_word2idx.get(word.strip(',." ;:)(][?!'), 0) for word in sentence.split()] for sentence in english_sentences]
	y = [[spanish_word2idx.get(word.strip(',." ;:)(][?!'), 0) for word in sentence.split()] for sentence in spanish_sentences]

	X = []
	Y = []
	for i in range(len(x)):
		n1 = len(x[i])
		n2 = len(y[i])
		n = n1 if n1 < n2 else n2
		if abs(n1 - n2) <= 0.3 * n:
			if n1 <= 15 and n2 <= 15:
				X.append(x[i])
				Y.append(y[i])

	return X, Y, english_word2idx, english_idx2word, english_vocab, spanish_word2idx, spanish_idx2word, spanish_vocab

def save_dataset(file_path, obj):
	with open(file_path, 'wb') as f:
		pickle.dump(obj, f, -1)

def read_dataset(file_path):
	with open(file_path, 'rb') as f:
		return pickle.load(f, encoding='latin1')

def main():
	english_sentences = read_sentences('data/data.en')
	spanish_sentences = read_sentences('data/data.es')
	save_dataset('./data.pkl', create_dataset(english_sentences, spanish_sentences))

if __name__ == '__main__':
	main()