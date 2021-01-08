import os

HOME_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    # hive
    hive_host = '172.26.171.58'
    hive_port = '10000'
    hive_user = 'root'
    hive_db = 'pv_ev'
    # data_dir
    data_dir = os.path.join(HOME_DIR, 'data')
    # dict_dir
    dict_dir = os.path.join(HOME_DIR, 'dict')
    # 关键词词典
    keywords_file_path = os.path.join(dict_dir, 'kwords.dict')
    # 车型车系词典
    t_veco_brand_series_file_path = os.path.join(dict_dir, 't_veco_brand_series.xlsx')
    # BERT配置
    bert_config_path = '/dev/DATA_BDP/BERT/chinese_L-12_H-768_A-12/bert_config.json'
    bert_checkpoint_path = '/dev/DATA_BDP/BERT/chinese_L-12_H-768_A-12/bert_model.ckpt'
    bert_dict_path = '/dev/DATA_BDP/BERT/chinese_L-12_H-768_A-12/vocab.txt'
    # albert配置
    albert_config_path = '/dev/DATA_BDP/ALBERT/albert_tiny/albert_config_tiny_g.json'
    albert_checkpoint_path = '/dev/DATA_BDP/ALBERT/albert_tiny/albert_model.ckpt'
    albert_dict_path = '/dev/DATA_BDP/ALBERT/albert_tiny/vocab.txt'
    # 中间文件暂存路径
    # 舆情日报本地文件
    tcs_file_path = os.path.join(data_dir, 'tcs_download.csv')
    # 文本列
    textcolumn = 'sentence'
    # 舆情日报分句文件
    tcs_seg_file_path = os.path.join(data_dir, 'tcs_seg.csv')
    # 舆情日报分句文件(带关键词)
    tcs_seg_with_keywords_file_path = os.path.join(data_dir, 'tcs_seg_with_keywords.csv')
    # 舆情日报分句文件(带车型识别)
    tcs_seg_with_keywords_ner_file_path = os.path.join(data_dir, 'tcs_seg_with_keywords_ner.csv')
    # 舆情日报分句文件(带标签)
    tcs_seg_with_keywords_label_file_path = os.path.join(data_dir, 'tcs_seg_with_keywords_label.csv')
    # 舆情日报分句文件(带车型、带标签)
    tcs_seg_with_ner_label_file_path = os.path.join(data_dir, 'tcs_seg_with_ner_label.csv')
    # 舆情日报分句文件(带车型、带标签、带日产)
    tcs_seg_with_ner_label_pv_file_path = os.path.join(data_dir, 'tcs_seg_with_ner_label_pv.csv')
    # tcs_upload
    tcs_upload_file_path = os.path.join(data_dir, 'tcs_upload.csv')
    # tcs_upload_hive
    tcs_upload_hive_file_path = 'tcs_upload_hive.csv'


class LabelClassificationConfig(Config):
    # 数据集
    label_data_dir = os.path.join(HOME_DIR, "tcs_multilabel_model", "dataset")
    train_file = os.path.join(label_data_dir, "train.json")
    valid_file = os.path.join(label_data_dir, "valid.json")
    test_file  = os.path.join(label_data_dir, "test.json")
    labelfile  = os.path.join(label_data_dir, "TCSLabel.xlsx")
    # 模型
    ckpt_dir = os.path.join(HOME_DIR, "tcs_multilabel_model", "ckpt")
    best_model_path = os.path.join(ckpt_dir, "weights.hdf5")
    
class CarNerConfig(Config):
    # 数据集
    data_dir = os.path.join(HOME_DIR, "car_ner_model", "dataset")
    train_file = os.path.join(data_dir, "train_data.txt")
    valid_file = os.path.join(data_dir, "valid_data.txt")
    test_file  = os.path.join(data_dir, "test_data.txt")
    # 模型
    ckpt_dir = os.path.join(HOME_DIR, "car_ner_model", "ckpt")
    best_model_path = os.path.join(ckpt_dir, "weights.hdf5")
