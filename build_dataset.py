# external libraries
import os
from tqdm import tqdm
import json
import urllib.request

from utils import tokenizer, clean_text, word_tokenize, sent_tokenize, convert_idx

MAX_SENTENCE_LENGTH = 100
MAX_QUESTION_LENGTH = 100

class SquadPreprocessor:

    def __init__(self, save_dir):
        self.save_dir = save_dir

    def load_data(self, file_name):
        filepath = os.path.join(self.data_dir, file_name)
        with open(filepath) as f:
            return json.load(f)

    def split_sentence_question(self,filename,data_type): 
        data = self.load_data(filename)
        with open(os.path.join(self.save_dir, data_type + '.sentence'), 'w', encoding="utf-8") as sentence_file,\
             open(os.path.join(self.save_dir, data_type + '.question'), 'w', encoding="utf-8") as question_file:
            
            artilces = data['data']
            for article in tqdm(artilces):
                paragraphs = article['paragraphs']
                for paragraph in paragraphs:
                    context = paragraph['context']

                    context = clean_text(context)
                    context_tokens = word_tokenize(context)                    
                    context_sentences = sent_tokenize(context)

                    spans = convert_idx(context, context_tokens)
                    num_tokens = 0
                    first_token_sentence = []
                    for sentence in context_sentences:
                        first_token_sentence.append(num_tokens)
                        num_tokens += len(sentence)

                    question_and_answer_list = paragraph['qa']
                    for question_and_answer in question_and_answer_list:
                        question = question_and_answer['question']
                        question = clean_text(question)
                        question_tokens = word_tokenize(question)

                        if not question_and_answer['answer'] : continue
                        answer = question_and_answer['answers']['0']
                        answer_text = answer['text']
                        answer_text = clean_text(answer_text)
                        answer_tokens = word_tokenize(answer)
                        answer_start = answer['answer_start']
                        answer_stop = answer_start + len(answer)

                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_stop <= span[0] or answer_start >= span[1]):
                                    answer_span.append(idx)
                        if not answer_span: continue
                        
                        sentence_tokens = []
                        for idx, start in enumerate(first_token_sentence):
                            if answer_span[0] >= start:
                                sentence_tokens = context_sentences[idx]
                                answer_sentence_span = [span - start for span in answer_span]
                            else:
                                break
                        if not sentence_tokens:
                            print("Sentence cannot be found")
                            raise Exception()

                        if len(sentence_tokens) >= MAX_SENTENCE_LENGTH or len(question) >= MAX_QUESTION_LENGTH

                        
                        sentence_file.write(" ".join([token + u"￨" + "1" if idx in answer_sentence_span else token + u"￨" + "0" for idx, token in enumerate(sentence_tokens)]) + "\n")
                        question_file.write(" ".join([token for token in question_tokens]) + "\n")

    def preprocess(self,train_path, dev_path,test_path):

        #self.split_data(self.train_filename)
        #self.split_data(self.dev_filename)


if __name__ == "__main__":
    squad_train_filename = "train.json"
    squad_dev_filename = "dev.json"
    squad_test_filename = "test.json"
    save_dir = "data/squad/preprocessed"

    preprocessor = SquadPreprocessor(save_dir=save_dir)
    preprocessor.preprocess(train_path=squad_train_filename, test_path=squad_test_filename,
      dev_path=squad_dev_filename)
