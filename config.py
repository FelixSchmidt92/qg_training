# data directories
squad_data_dir = "data/squad/"
out_dir = "data/qg/"
train_dir = squad_data_dir + "train/"
dev_dir = squad_data_dir + "dev/"

# preprocessing values
paragraph = False
min_len_context = 5
max_len_context = 100 if not paragraph else 1000
min_len_question = 5
max_len_question = 20
cuda = False
