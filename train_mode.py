from car_ner import model, tokenizer, NER, evaluate
from car_ner import maxlen, id2label, label2id, num_labels

maxlen = 256
epochs = 10
batch_size = 32
learing_rate = 1e-5  # bert_layers越小，学习率应该要越大
crf_lr_multiplier = 1000  # 必要时扩大CRF层的学习率

def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d, last_flag = [], ''
            for c in l.split('\n'):
                char, this_flag = c.split('\t')
                this_flag = this_flag.upper()
                if this_flag != 'O':
                    this_flag += '-CAR'
                if this_flag == 'O' and last_flag == 'O':
                    d[-1][0] += char
                elif this_flag == 'O' and last_flag != 'O':
                    d.append([char, 'O'])
                elif this_flag[:1] == 'B':
                    d.append([char, this_flag[2:]])

                else:
                    d[-1][0] += char
                last_flag = this_flag
            D.append(d)
    return D


# 标注数据
train_data = load_data('./train_data.txt')
valid_data = load_data('./valid_data.txt')
test_data = load_data('./test_data.txt')

with open('./veco_list.txt', 'r', encoding='utf-8') as f:
    veco_series_list = [row.strip() for row in f if row.strip()]

class data_generator(DataGenerator):
    """数据生成器
    """
    def __init__(self, *args, enhance=False, frac=0.2, **kw):
        super().__init__(*args, **kw)
        self.enhance = enhance
        self.frac = frac

    def __iter__(self, random_=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, item in self.sample(random_):
            token_ids, labels = [tokenizer._token_start_id], [0]
            for w, l in item:
                if self.enhance and (l != 'O') and (random.random() < self.frac):
                    w = random.choice(veco_series_list)
                w_token_ids = tokenizer.encode(w)[0][1:-1]
                if len(token_ids) + len(w_token_ids) < maxlen:
                    token_ids += w_token_ids
                    if l == 'O':
                        labels += [0] * len(w_token_ids)
                    else:
                        B = label2id[l] * 2 + 1
                        I = label2id[l] * 2 + 2
                        labels += ([B] + [I] * (len(w_token_ids) - 1))
                else:
                    break
            token_ids += [tokenizer._token_end_id]
            labels += [0]
            segment_ids = [0] * len(token_ids)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []




best_val_f1 = 0
val_history, test_history = [], []
class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):
        global best_val_f1, val_history, test_history
        trans = K.eval(CRF.trans)
        NER.trans = trans
        print(NER.trans)
        print('Valid:')
        f1, precision, recall = evaluate(valid_data, location=True)
        val_history.append([f1, precision, recall])
        # 保存最优
        if f1 >= best_val_f1:
            best_val_f1 = f1
            model.save_weights('./checkpoints/best_model.weights')
        print('f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' % (f1, precision, recall, best_val_f1))
        print('Test:')
        f1, precision, recall = evaluate(test_data, location=True)
        test_history.append([f1, precision, recall])
        print('f1: %.5f, precision: %.5f, recall: %.5f\n' % (f1, precision, recall))

        
# model.summary()

model.compile(
    loss=CRF.sparse_loss,
    optimizer=Adam(learing_rate),
    metrics=[CRF.sparse_accuracy]
)


if __name__ == '__main__':
    evaluator = Evaluator()
    print('*'*20)
    print('*'*20)
    print('Step 1:')
    epochs = 3
    best_frac = 0
    tmp_val_f1 = 0
    for frac in [0, 0.1, 0.2, 0.3]:
        train_generator = data_generator(train_data, batch_size, enhance=True, frac=frac)
        model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epochs,
            callbacks=[evaluator]
        )
        if best_val_f1 > tmp_val_f1:
            tmp_val_f1 = best_val_f1
            best_frac = frac

    print('*'*20)
    print('*'*20)
    print('Step 2:')
    best_frac = 0.4
    print('Best random Frac:', best_frac)
    epochs = 20
    train_generator = data_generator(train_data, batch_size, enhance=True, frac=best_frac)
    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator]
    )
    
    with open('valid_history.txt', 'w') as f:
        f.write('\n'.join(['{}\t{}\t{}'.format(*row) for row in val_history]))

    with open('test_history.txt', 'w') as f:
        f.write('\n'.join(['{}\t{}\t{}'.format(*row) for row in test_history]))

    corpus = ['']

    for text in corpus:
        print(NER.recognize(text, location=True))
# else:
#     model.load_weights('./best_model.weights')
    # print('Test:')
    # f1, precision, recall = evaluate(test_data)
    # print('f1: %.5f, precision: %.5f, recall: %.5f\n' % (f1, precision, recall))

