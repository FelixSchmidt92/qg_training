# experiment ID
#exp = "qg-1"

# data directories
squad_data_dir = "data/squad/"
out_dir = "data/qg/"
train_dir = squad_data_dir + "train/"
dev_dir = squad_data_dir + "dev/"

# model paths
#spacy_en = "/Users/gdamien/Data/spacy/en_core_web_sm-2.0.0/en_core_web_sm/en_core_web_sm-2.0.0"
#glove = "/Users/gdamien/Data/glove.6B/"
#squad_models = "/Users/gdamien/Data/squad/models/"

# preprocessing values
paragraph = False
min_len_context = 5
max_len_context = 100 if not paragraph else 1000
min_len_question = 5
max_len_question = 20
cuda = False
#word_embedding_size = 300
#answer_embedding_size = 2
#in_vocab_size = 45000
#out_vocab_size = 28000
