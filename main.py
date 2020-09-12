from core.config import cfg
from utils.io import load_conversation


q, a = load_conversation(cfg.TRAIN.DATA_PATH)
# print(q)
# print(a)