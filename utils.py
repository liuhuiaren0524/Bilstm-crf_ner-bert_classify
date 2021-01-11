import json
import pandas as pd
from collections import OrderedDict

from config import Config as CONFIG
import logging

LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(pathname)s %(message)s "#配置输出日志格式
DATE_FORMAT = '%Y-%m-%d  %H:%M:%S %a ' #配置输出时间的格式，注意月份和天数不要搞乱了
logging.basicConfig(level=logging.INFO,
                    format=LOG_FORMAT,
                    datefmt = DATE_FORMAT ,
                    # filename=r"d:\test\test.log" #有了filename参数就不会直接输出显示到控制台，而是直接写入文件
                    )


def get_hive_connection():
    from pyhive import hive
    "连接hive数据库"
    conn = hive.Connection(host=CONFIG.hive_host,
                           port=CONFIG.hive_port,
                           username=CONFIG.hive_user,
                           database=CONFIG.hive_db)
    return conn


def load_data_from_hive(conn, sql):
    "从hive数据库下载数据，并以dataframe格式返回"
    with conn.cursor() as cursor:
        cursor.execute("SET hive.auto.convert.join=false")  # 关闭自动MAPJOIN转换操作)
        cursor.execute("SET hive.ignore.mapjoin.hint=false")  # 不忽略MAPJOIN标记
        cursor.execute(sql)
        data = cursor.fetchall()
        columns = [col[0].split('.')[-1].lower() for col in cursor.description]
        df = pd.DataFrame(list(data), columns=columns)
    return df

#from config import CONFIG

def retro_dictify(frame):
    "根据dataframe嵌套创建有序字典"
    d = OrderedDict()
    for row in frame.values:
        here = d
        for elem in row[:-2]:
            if elem not in here:
                here[elem] = OrderedDict()
            here = here[elem]
        if row[-2] in here:
            print('重复键：{}'.format(row[-2]))
        else:
            here[row[-2]] = row[-1]
    return d


def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as fp:
        fp.write(json.dumps(data, ensure_ascii=False, indent=4))
        

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
