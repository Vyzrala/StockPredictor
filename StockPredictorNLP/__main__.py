from .tools import preprocess_raw_datasets, read_files_and_yahoo


path = 'D:\Programing projects\Thesis datasets'
all, yahoo = read_files_and_yahoo(path)

nlp_data = preprocess_raw_datasets(all, yahoo)
print(nlp_data)

for k, v in nlp_data.items():
    v.to_csv(path+'\\'+k+'.csv')