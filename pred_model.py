import sys, getopt
import pandas as pd
from tqdm import tqdm

from car_ner import model, tokenizer, NER
from car_ner import maxlen, id2label, label2id, num_labels

model.load_weights('./best_model.weights')

                
def bert_token(data, maxlen):
    token_ids, segment_ids = [], []
    for i, text in enumerate(data):
        t_ids, s_ids = tokenizer.encode(text, maxlen=maxlen) # tokenizer 需要提前定义
        token_ids.append(t_ids)
        segment_ids.append(s_ids)
    token_ids = sequence_padding(token_ids)
    segment_ids = sequence_padding(segment_ids)
    return [token_ids, segment_ids]


def main():
   argv = sys.argv[1:]
   inputfile = ''
   textcolumn = ''
   try:
      opts, args = getopt.getopt(argv,"hi:o:c:",["ifile=","ofile=","cloumn="])
   except getopt.GetoptError:
      print ('test.py -i <inputfile> -o <outputfile> -c <textcolumn>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print ('test.py -i <inputfile> -o <outputfile> -c <textcolumn>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-c", "--cloumn"):
         textcolumn = arg
   print ('输入的文件为：', inputfile)
   print ('文本所在列为: ', textcolumn)
    
   outputfile = inputfile.replace('.', '_car.', 1)
 
   ftype = inputfile.split('.')[-1]
   if ftype == 'csv':
      df = pd.read_csv(inputfile)
   elif ftype == 'xlsx':
      df = pd.read_excel(inputfile)
   else:
      raise Exception('输入文件格式不正确，合理的输入文件格式为："csv","xlsx"') 
   
   df = df.head(100)
   print(df.head())
    
   pred = []
   for text in df[textcolumn].values:
        pred.append(NER.recognize(text, location=True))
   
   df[textcolumn+'_car'] = pred
   df.to_csv(outputfile, index=False, encoding='utf-8-sig')

if __name__ == "__main__":
   main()
