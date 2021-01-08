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
                tag_.append(label)
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
valid_data = list(zip(test_datas, test_labels))

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
EPOCHS = 50
BATCH_SIZE = 64
EMBED_DIM = 128
HIDDEN_SIZE = 64
MAX_LEN = 100
VOCAB_SIZE = len(vocab2idx)
CLASS_NUMS = len(label2idx)

veco_brand_series = pd.read_excel('t_veco_brand_series.xlsx', na_values='NA')
veco_brand_series = veco_brand_series[['BRAND', 'CAR_SERIES', 'KEY_WORD']].dropna().drop_duplicates()
veco_series_list = veco_brand_series.CAR_SERIES.str.lower().values.tolist()

train_D = data_generator(train_data, max_len=MAX_LEN, class_num=3, batch_size=BATCH_SIZE, enhance=True, veco_series_list=veco_series_list)

tmp = train_D.__iter__()
class SetLearningRate:
    """层的一个包装，用来设置当前层的学习率
    """

    def __init__(self, layer, lamb, is_ada=False):
        self.layer = layer
        self.lamb = lamb # 学习率比例
        self.is_ada = is_ada # 是否自适应学习率优化器

    def __call__(self, inputs):
        with K.name_scope(self.layer.name):
            if not self.layer.built:
                input_shape = K.int_shape(inputs)
                self.layer.build(input_shape)
                self.layer.built = True
                if self.layer._initial_weights is not None:
                    self.layer.set_weights(self.layer._initial_weights)
        for key in ['kernel', 'bias', 'embeddings', 'depthwise_kernel', 'pointwise_kernel', 'recurrent_kernel', 'gamma', 'beta']:
            if hasattr(self.layer, key):
                weight = getattr(self.layer, key)
                if self.is_ada:
                    lamb = self.lamb # 自适应学习率优化器直接保持lamb比例
                else:
                    lamb = self.lamb**0.5 # SGD（包括动量加速），lamb要开平方
                K.set_value(weight, K.eval(weight) / lamb) # 更改初始化
                setattr(self.layer, key, weight * lamb) # 按比例替换
        return self.layer(inputs)
from keras.layers import Input, Embedding, Bidirectional, LSTM, Masking, TimeDistributed,Dense
from keras.models import Model, load_model
from keras.optimizers import Adam, SGD, RMSprop,Adagrad,Adadelta
from keras.regularizers import l1_l2
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

def BiLstmCRF_FC(max_len, vocab_size, embed_dim, class_nums):
    l1, l2 = 0.0001, 0.001
    inputs = Input(shape=(max_len,), dtype='int32')
    x = Masking(mask_value=0)(inputs)
    x = Embedding(vocab_size, embed_dim, mask_zero=True)(x)
    x = SetLearningRate(Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l1_l2(l1=l1, l2=l2))), 0.1, True)(x)
    x = SetLearningRate(Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l1_l2(l1=l1, l2=l2))), 0.1, True)(x)
    x = SetLearningRate(Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l1_l2(l1=l1, l2=l2))), 0.1, True)(x)
    x = TimeDistributed(Dense(class_nums))(x)
    outputs = CRF(class_nums)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model
def BiLstmCRF(max_len, vocab_size, embed_dim, class_nums):
    inputs = Input(shape=(max_len,), dtype='int32')
    x = Masking(mask_value=0)(inputs)
    x = Embedding(vocab_size, embed_dim, mask_zero=True)(x)
    x = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l1_l2(l1=0.01, l2=0.01)))(x)
    x = TimeDistributed(Dense(class_nums))(x)
    outputs = CRF(class_nums)(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model
	
	model_path = "./checkpoint/ch_ner_model.h5" # 模型文件


#样本总数
SAMPLE_COUNT = 2999

# Number of warmup epochs.
WARMUP_EPOCH = 3

# Base learning rate after warmup.
LEARNING_RATE_BASE = 0.001

total_steps = int(EPOCHS * SAMPLE_COUNT / BATCH_SIZE)
print('total_steps:', total_steps)

# Compute the number of warmup batches.
warmup_steps = int(WARMUP_EPOCH * SAMPLE_COUNT / BATCH_SIZE)
print('warmup_steps:', warmup_steps)
# Create the Reduce
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
# Create the Checkpointer
checkpointer = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True)

# Create the Earlystopping
earlystop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

# Create the Learning rate scheduler.
warm_up_lr = WarmUpCosineDecayScheduler(learning_rate_base=LEARNING_RATE_BASE,
                                        total_steps=total_steps,
                                        warmup_learning_rate=4e-06,
                                        warmup_steps=warmup_steps,
                                        hold_base_rate_steps=5,
                                        )


opt = Adam()
model.compile(loss=crf_loss, optimizer=opt, metrics=[crf_viterbi_accuracy])
# 训练模型
history = model.fit_generator(train_D.__iter__(), 
                              steps_per_epoch=len(train_D), 
                              epochs=EPOCHS, 
                              verbose=1, 
                              validation_data=valid_D.__iter__(), 
                              validation_steps=len(valid_D),
#                               callbacks=[checkpointer, warm_up_lr, earlystop],
                              callbacks=[checkpointer, earlystop, reduce_lr],
                   )
# 测试模型效果
model = load_model(model_path, custom_objects={'CRF': CRF, 'crf_loss':crf_loss, 'crf_viterbi_accuracy':crf_viterbi_accuracy})
score = model.evaluate(test_datas, test_labels, batch_size=BATCH_SIZE)
print(model.metrics_names)
print(score)

# 保存模型
# model.save(model_path)