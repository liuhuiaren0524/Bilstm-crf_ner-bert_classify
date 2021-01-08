#! -*- coding: utf-8 -*-
# 用CRF做中文命名实体识别 苏剑林 bert4keras
# 数据集 http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
# 实测验证集的F1可以到96.18%，测试集的F1可以到95.35%
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, ViterbiDecoder, to_array
from bert4keras.layers import ConditionalRandomField
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm

train_data_dir = r'C:/Users/ly-lhr.PVI/车型识别模型/模型构建/train_data/'
char_vocab_path = train_data_dir + "char_vocabs.txt" # 字典文件
train_data_path = train_data_dir + 'train_data.txt'  # 训练数据
valid_data_path = train_data_dir + 'valid_data.txt'  # 验证数据
test_data_path =  train_data_dir + 'test_data.txt'   # 测试数据

special_words = ['<PAD>', '<UNK>'] # 特殊词表示

# "BIO"标记的标签
label2idx = {"O": 0, "B": 1, "I": 2}
# 索引和BIO标签对应
idx2label = {idx: label for label, idx in label2idx.items()}

# 读取字符词典文件
with open(char_vocab_path, "r", encoding="utf8") as fo:
    char_vocabs = [line.strip() for line in fo if line.strip()]  # 空格当成未知字符
char_vocabs = special_words + char_vocabs

# 字符和索引编号对应
idx2vocab = {idx: char for idx, char in enumerate(char_vocabs)}
vocab2idx = {char: idx for idx, char in idx2vocab.items()}

# 读取数据集语料
def read_corpus(corpus_path, vocab2idx, label2idx):
    datas, labels = [], []
    with open(corpus_path, encoding='utf-8') as fr:
        sent_, tag_ = '', []
        for line in fr:
            if line != '\n':
                [char, label] = line.rstrip().split('\t')
                sent_+=char
                tag_.append(label.upper())
            else:
#                 sent_ids = [vocab2idx[char] if char in vocab2idx else vocab2idx['<UNK>'] for char in sent_]
#                 tag_ids = [label2idx[label] if label in label2idx else 0 for label in tag_]
                datas.append(sent_)
                labels.append(tag_)
                sent_, tag_ = '', []
    return datas, labels

# 加载训练集
train_datas, train_labels = read_corpus(train_data_path, vocab2idx, label2idx)
train_data = list(zip(train_datas, train_labels))
# 加载验证集
valid_datas, valid_labels = read_corpus(valid_data_path, vocab2idx, label2idx)
valid_data = list(zip(valid_datas, valid_labels))
# 加载测试集
test_datas, test_labels =   read_corpus(test_data_path,  vocab2idx, label2idx)
test_data = list(zip(test_datas, test_labels))

import numpy as np
import pandas as pd
import random
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

class data_generator:
    def __init__(self, data, max_len, class_num, batch_size=64, enhance=False, veco_series_list=None):
        self.data = data
        self.max_len = max_len
        self.class_num = class_num
        self.enhance = enhance
        self.veco_series_list = veco_series_list
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1
    def __len__(self):
        return self.steps
    def __iter__(self):
        while True:
            idxs = list(range(len(self.data)))
            np.random.shuffle(idxs)
            T, L, = [], []
            for i in idxs:
                d = self.data[i]
                text = d[0]
                label = d[1]
#                 print('old_text', text)
                if self.enhance:
                    text, label = self.replace_tag(text, label)
#                 print('new_text:', text)
                T.append([vocab2idx[c] if c in vocab2idx else vocab2idx['<UNK>'] for c in text])
                L.append([label2idx[l] if l in label2idx else 0 for l in label])
                if len(T) == self.batch_size or i == idxs[-1]:
                    T = pad_sequences(T, maxlen=self.max_len, padding='post', truncating='post')
                    L = pad_sequences(L, maxlen=self.max_len, padding='post', truncating='post')
                    L = to_categorical(L, self.class_num)
                    yield T, L
                    T, L = [], []
                    
                    
    def replace_tag(self, sentence, label, frac=0.5):
        """
        把原始文本中的目标实体替换成同类其他实体
        @sentence：原始文本（“日产轩逸真的很棒”）
        @label：文本序列标签（['O','O','B','I','O','O','O','O']）
        @veco_series_list：标签字典列表(['轩逸', '天籁', '逍客'])
        return: “日产天籁真的很棒” 或者 “日产逍客真的很棒”
        """
        tmp_tag = ''
        new_sentence = ''
        new_label = []
        start = False
        for c, l in zip(sentence, label):
            if start == True:
                if l == 'B': # 结束当前标签，开始下一个标签
                    if random.random() > frac:
                        tmp_tag = random.choice(self.veco_series_list)
                    new_sentence += tmp_tag
                    new_label.extend(['B']+['I']*(len(tmp_tag)-1))
                    start = True
                    tmp_tag = ''
                elif l == 'I': # 跳过
                    tmp_tag += c
                else: # 结束当前标签
                    if random.random() > frac:
                        tmp_tag = random.choice(self.veco_series_list)
                    new_sentence += tmp_tag
                    new_label.extend(['B']+['I']*(len(tmp_tag)-1))
                    new_sentence += c
                    new_label.append(l)
                    start = False
                    tmp_tag = ''
            else:
                if l == 'B': # 开始一个标签
                    start=True
                    tmp_tag += c
                elif l == 'I': # 标签以‘I’开始，有误，跳过
                    new_sentence += c
                    new_label.append('O')
                else: # 其他标签，跳过
                    new_sentence += c
                    new_label.append(l)
        return new_sentence, new_label
def f1(*args, c=0, **kw):
    print('c =', c, 'args =', args, 'kw =', kw)

f1(1,2,3,4,5,d=111,c=55, e='jd')

import time

def BiLstmCRF(max_len, vocab_size, embed_dim, class_nums):
    inputs = Input(shape=(max_len,), dtype='int32')
    x = Masking(mask_value=0)(inputs)
    x = Embedding(vocab_size, embed_dim, mask_zero=True)(x)
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))(x)
    x = TimeDistributed(Dense(class_nums))(x)
    output = CRF(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model



class NamedEntityRecognizer(ViterbiDecoder):
    """命名实体识别器
    """
    def recognize(self, text):
#         tokens = tokenizer.tokenize(text)
#         while len(tokens) > 512:
#             tokens.pop(-2)
#         mapping = tokenizer.rematch(text, tokens)
#         token_ids = tokenizer.tokens_to_ids(tokens)
#         segment_ids = [0] * len(token_ids)
#         token_ids, segment_ids = to_array([token_ids], [segment_ids])
        sent2id = [vocab2idx[word] if word in vocab2idx else vocab2idx['<UNK>'] for word in text]
        sent2input = np.array([sent2id[:MAX_LEN] + [0] * (MAX_LEN-len(sent2id))])
        nodes = model.predict([sent2input])[0]
        labels = self.decode(nodes)
        entities, starting = [], False
        for i, label in enumerate(labels):
            if label > 0:
                if label % 2 == 1:
                    starting = True
                    entities.append([[i], label])
                elif starting:
                    entities[-1][0].append(i)
                else:
                    starting = False
            else:
                starting = False

        return ['{}({}:{})'.format(text[w[0]:w[-1] + 1], w[0], w[-1]+1) for w, l in entities]

def extr(text, label):
    d, last_flag = [], ''
    for i, [char, this_flag] in enumerate(zip(text, label)):
        if this_flag == 'O' and last_flag == 'O':
            d[-1][0] += char
            d[-1][-1].append(i)
        elif this_flag == 'O' and last_flag != 'O':
            d.append([char, 'O', [i]])
        elif this_flag[:1] == 'B':
            d.append([char, this_flag[2:], [i]])
        else:
            d[-1][0] += char
            d[-1][-1].append(i)
        last_flag = this_flag
    return d

def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data):
        text = d[0]
        R = set(NER.recognize(text))
        T = set(['{}({}:{})'.format(i[0], i[-1][0], i[-1][-1]+1) for i in extr(*d) if i[1] != 'O'])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall



class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        trans = K.eval(CRF.trans)
        NER.trans = trans
        print('NER.trans:\n', NER.trans)
        time.sleep(0.1)
        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('./best_model.weights')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        time.sleep(0.1)
        f1, precision, recall = evaluate(test_data)
        print(
            'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
        )
from keras.layers import Input, Embedding, Bidirectional, LSTM, Masking, TimeDistributed,Dense
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD, RMSprop,Adagrad,Adadelta
from keras.regularizers import l1_l2
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

EPOCHS = 100
BATCH_SIZE = 64
EMBED_DIM = 128
HIDDEN_SIZE = 64
MAX_LEN = 100
VOCAB_SIZE = len(vocab2idx)
CLASS_NUMS = len(label2idx)

veco_brand_series = pd.read_excel('t_veco_brand_series.xlsx', na_values='NA')
veco_brand_series = veco_brand_series[['BRAND', 'CAR_SERIES', 'KEY_WORD']].dropna().drop_duplicates()
veco_series_list = veco_brand_series.CAR_SERIES.str.lower().values.tolist()

with open('veco_series_list.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(veco_series_list))
	
veco_brand_series = pd.read_excel('t_veco_brand_series.xlsx', na_values='NA')
veco_brand_series = veco_brand_series[['BRAND', 'CAR_SERIES', 'KEY_WORD']].dropna().drop_duplicates()
veco_series_list = veco_brand_series.CAR_SERIES.str.lower().values.tolist()

train_D = data_generator(train_data, max_len=MAX_LEN, class_num=3, batch_size=BATCH_SIZE, enhance=True, veco_series_list=veco_series_list)
# valid_D = data_generator(valid_data, max_len=MAX_LEN, class_num=3, batch_size=BATCH_SIZE, enhance=False, veco_series_list=None)

# test_datas = [[vocab2idx[c] if c in vocab2idx else vocab2idx['<UNK>'] for c in text] for text in test_datas]
# test_labels= [[label2idx[l] if l in label2idx else 0 for l in label] for label in test_labels]
# test_datas = pad_sequences(test_datas, maxlen=MAX_LEN)
# test_labels = pad_sequences(test_labels, maxlen=MAX_LEN)
# test_labels = to_categorical(test_labels, CLASS_NUMS)

learing_rate = 0.001
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率
CRF = ConditionalRandomField(lr_multiplier=crf_lr_multiplier)


inputs = Input(shape=(MAX_LEN,), dtype='int32')
x = Masking(mask_value=0)(inputs)
x = Embedding(VOCAB_SIZE, EMBED_DIM, mask_zero=True)(x)
x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))(x)
x = TimeDistributed(Dense(CLASS_NUMS))(x)
outputs = CRF(x)
model = Model(inputs=inputs, outputs=outputs)

model.summary()

model.compile(
    loss=CRF.dense_loss,
    optimizer=Adam(learing_rate),
    metrics=[CRF.dense_accuracy]
)

NER = NamedEntityRecognizer(trans=K.eval(CRF.trans), starts=[0], ends=[0])

# Create the Reduce
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

evaluator = Evaluator()

history = model.fit_generator(train_D.__iter__(), 
                              steps_per_epoch=len(train_D), 
                              epochs=EPOCHS, 
                              verbose=1, 
                              callbacks=[evaluator, reduce_lr],
                   )
#model.load_weights('./best_model.weights')