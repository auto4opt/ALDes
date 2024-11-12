from conf import *
from util.data_loader import DataLoader
from util.tokenizer import Tokenizer


src_pad_idx = None
trg_pad_idx = None
trg_sos_idx = None
enc_voc_size = 100
dec_voc_size = 30