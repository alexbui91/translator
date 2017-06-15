import pickle
import properties
import utils
import argparse
from nltk.tokenize import RegexpTokenizer
from os import path


def main(data_path):
    if path.exists(data_path + "/dict_char_en.pkl"):
        dict_char_en = utils.load_file(data_path + "/dict_char_en.pkl")
    else:
        dict_char_en = generateCharacterDict(properties.en_char)
        utils.save_file(data_path + "/dict_char_en.pkl", dict_char_en)
    if path.exists(data_path + "/dict_char_vi.pkl"):
        dict_char_vi = utils.load_file(data_path + "/dict_char_vi.pkl")
    else:
        dict_char_vi = generateCharacterDict(properties.vi_char)
        utils.save_file(data_path + "/dict_char_vi.pkl", dict_char_vi)
    if path.exists(data_path + "/dict_en.pkl"):
        dict_en = utils.load_file(data_path + "/dict_en.pkl", True)
    else:
        dict_en = build_dictionary(data_path, properties.vocab_en)
        utils.save_file(data_path + "/dict_en.pkl", dict_en, True)
    if path.exists(data_path + "/dict_vi.pkl"):
        dict_vi = utils.load_file(data_path + "/dict_vi.pkl", True)
    else:
        dict_vi = build_dictionary(data_path, properties.vocab_vi)
        utils.save_file(data_path + "/dict_vi.pkl", dict_vi, True)
    
    dataset_en, unknown_en = map_sentence_idx(data_path + "/" + properties.train_en, dict_en, dict_char_en)
    dataset_vi, unknown_vi = map_sentence_idx(data_path + "/" + properties.train_vi, dict_vi, dict_char_vi)
    utils.save_file(data_path + "/dataset_en.pkl", (dataset_en, unknown_en))
    utils.save_file(data_path + "/dataset_vi.pkl", (dataset_vi, unknown_vi))


def generateCharacterDict(lang):
    char_set = ['<unk>'] + lang + properties.num_char + properties.spec_char
    char_set.sort()
    char_dict = dict()
    for index, c in enumerate(char_set):
        char_dict[c] = index
    return char_dict


def build_dictionary(data_path, vocab_path):
    vocab = utils.load_file_lines(data_path + "/" + vocab_path)
    vocab_dict = build_dictionary_from_list(vocab)
    return vocab_dict


def map_sentence_idx(train_path, vocab_dict, char_dict):
    dataset = utils.load_file_lines(train_path)
    dataset_idx = list()
    unknown_dataset = dict()
    m = properties.batch_size * 2
    for s_idx, sent in enumerate(dataset):
        if s_idx > m:
            break
        sent_idx = list()
        unknowns = dict()
        sent_words = sent.lower().split(" ")
        sent_length = len(sent_words)
        for index in range(properties.max_len):
            if index < sent_length:
                word = sent_words[index]
                if word in vocab_dict:
                    sent_idx.append(vocab_dict[word])
                else:
                    sent_idx.append(0)
                    unknowns[index] = map_char_dict(word, char_dict)
        dataset_idx.append(sent_idx)
        if unknowns:
            unknown_dataset[s_idx] = unknowns
    return dataset_idx, unknown_dataset


def build_dictionary_from_list(data):
    vocabs = dict()
    # tokenizer = RegexpTokenizer(r'\w+')
    for index, value in enumerate(data):
        value = value.lower().replace('\n', '')
        vocabs[value] = index
    return vocabs


def map_char_dict(word, dict_char):
    char_idx = list()
    if word:
        for c in word:
            if c in dict_char:
                char_idx.append(c)
            else: 
                char_idx.append(0)
    return char_idx


def load_dataset(data_path):
    dataset, unknown = utils.load_file(data_path)
    return dataset, unknown


def load_char_dict(data_path):
    dict_char_en = utils.load_file(data_path + "/dict_char_en.pkl")
    dict_char_vi = utils.load_file(data_path + "/dict_char_vi.pkl")
    return dict_char_en, dict_char_vi


def load_dict(data_path):
    dict_en = utils.load_file(data_path + "/dict_en.pkl")
    dict_vi = utils.load_file(data_path + "/dict_vi.pkl")
    return dict_en, dict_vi


if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='Preprocess data')
    parser.add_argument('--path', type=str, default='./data')

    args = parser.parse_args()

    main(args.path)

