import torch

EMBEDDING_MODEL_PATH = r".\Models\models--sentence-transformers--multi-qa-mpnet-base-cos-v1\snapshots\29be89d5a0e6baddfea61c75e66bc82d7b97f07a"
QUESTION_ANSWERING_MODEL_PATH = r".\Models\models--deepset--roberta-base-squad2\snapshots\e84d19c1ab20d7a5c15407f6954cef5c25d7a261"
EMBEDDING_MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-cos-v1"
QUESTION_ANSWERING_MODEL_NAME = "deepset/roberta-base-squad2"
TOP_K = 5
SIMILARITY_THRESHOLD = 0.5
QUESTION = "When was Da Vinci born?"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")