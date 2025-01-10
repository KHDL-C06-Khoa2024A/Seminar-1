# [Seminar 1] Nhóm 6

:::info
:bulb: Lớp học: KHDL-C06-Khoa2024A 
:::

## :beginner: Thành viên


1. Nguyễn Quốc Huy - Phó Giám đốc Trung tâm  
2. Nguyễn Hải Đăng - Cán bộ  
3. Trương Nhật Thành - Cán bộ  

## :triangular_flag_on_post: Báo cáo

- Slide:
- Báo cáo:

## :pencil: Source code



### :small_blue_diamond: Dữ liệu:
Thư mục Data chứa các file trích xuất từ dataset VFND


```data_path = './Data/'```

Một số file CSV được trích xuất từ VFND:

`vn_news_226_tlfr.csv`: Tin tức và Facebook post, 2 trường: text (Với các bài báo là Tiêu đề + Nội dung, với Facebook post là nội dung của post đó) và label

`vn_news_223_tdlfr.csv`: 3 trường: text, domain và label

Nguồn: https://github.com/WhySchools/VFND-vietnamese-fake-news-datasets/tree/master

### :small_blue_diamond: File từ điển:

Đường dẫn đến các file từ điển

```dict_path = './Dictionaries'```

```
bi_gram_path = os.path.join(dict_path, 'bi_gram.txt')
tri_gram_path = os.path.join(dict_path, 'tri_gram.txt')
four_gram_path = os.path.join(dict_path, 'four_gram.txt')

stopword_path = os.path.join(dict_path, 'Stopwords_vi.txt')
Load các từ điển đó lên và chứa vào các Set() dữ liệu
```
```
bi_dict = load_file_with_newline(bi_gram_path)
tri_dict = load_file_with_newline(tri_gram_path)
four_dict = load_file_with_newline(four_gram_path)

stopwords = load_file_with_newline(stopword_path)
```
Đưa các từ trong từ điển về dạng chữ thường
```
bi_gram = [x.lower() for x in bi_dict]
tri_gram = [x.lower() for x in tri_dict]
four_gram = [x.lower() for x in four_dict]
```
### :small_blue_diamond: Thuật toán Longest Matching:
Thuật toán Longest Matching gộp các syllable token - tiếng - thành các từ được định nghĩa trong bộ từ điển. Input:

token: List của các syllable sau khi tách ra
bi_gram, tri_gram, four_gram: Các set() từ điển đã load ở trên

```python
def LongestMatching(token, bi_gram, tri_gram, four_gram):
    #token_len: Length of syllable list
    token_len = len(token)
    cur_id = 0
    word_list = []
    
    #done: True when cur_id reach 
    done = False
    
    while (cur_id < token_len) and (not done):
        cur_word = token[cur_id]
        if(cur_id >= token_len - 1):
            word_list.append(cur_word)
            done = True
        else:
            next_word = token[cur_id + 1]
            bi_word = " ".join([cur_word.lower(), next_word.lower()])
            if(cur_id >= token_len - 2):
                if bi_word in bi_gram:
                    word_list.append("_".join([cur_word, next_word]))
                    cur_id  = cur_id + 2
                else: 
                    word_list.append(cur_word)
                    cur_id  = cur_id + 1
                        
            else: 
                bi_next_word = token[cur_id + 2]
                tri_word = " ".join([bi_word, bi_next_word.lower()])
                if(cur_id >= token_len - 3):
                    if tri_word in tri_gram:
                        word_list.append("_".join([cur_word, next_word, bi_next_word]))
                        cur_id  = cur_id + 3
                    elif bi_word in bi_gram:
                        word_list.append("_".join([cur_word, next_word]))
                        cur_id = cur_id + 2
                    else:
                        word_list.append(cur_word)
                        cur_id = cur_id + 1
                else:
                    tri_next_word = token[cur_id + 3]
                    four_word = " ".join([tri_word, tri_next_word.lower()])
                    if four_word in four_gram:
                        word_list.append("_".join([cur_word, next_word, bi_next_word, tri_next_word]))
                        cur_id  = cur_id + 4
                    elif tri_word in tri_gram:
                        word_list.append("_".join([cur_word, next_word, bi_next_word]))
                        cur_id  = cur_id + 3
                    elif bi_word in bi_gram:
                        word_list.append("_".join([cur_word, next_word]))
                        cur_id = cur_id + 2
                    else:
                        word_list.append(cur_word)
                        cur_id = cur_id + 1
    return word_list
```
### :small_blue_diamond:  Text Preprocessing:
- Input: văn bản tin tức 
- Output: Văn bản của các cụm từ đã được gom cụm theo bộ từ điển:

Chuyển đổi text thành các âm tiết bằng hàm: `token_sylabling`

Dùng longest matching để chuyển các tiếng thành các từ dựa theo bộ từ điển
Loại bỏ stopword

```python
def text_preprocessing(text):
    token_list = token_sylabling(text)
    word_list = LongestMatching(token_list, bi_gram, tri_gram, four_gram)
    remove_stopword_list = remove_stopwords(word_list, stopwords)
    new_text = ' '.join(remove_stopword_list)
    return new_text
```
## Load và xử lý các file
### Load và xử lý vn_news_226_tlfr.csv
Load vn_news_226_tlfr.csv vào List các Dictionary


```python
newslist_1 = []

with open(os.path.join(data_path, csv_text_label_226), newline='', encoding = 'UTF-8') as csv_file:
    reader = csv.DictReader(csv_file)
    newslist_1 = list(reader)
```    
List các tin tức đã được tiền xử lý
```python
preprocessed_news_1 = []
for news in newslist_1:
    pre_news = dict()
    pre_news['text'] = text_preprocessing(news['text'])
    pre_news['label'] = news['label']
    preprocessed_news_1.append(pre_news)
```    
### Load và xử lý vn_news_223_tdlfr.csv

```python
newslist_2 = []

with open(os.path.join(data_path, csv_text_domain_label_223), newline='', encoding = 'UTF-8') as csv_file:
    reader = csv.DictReader(csv_file)
    newslist_2 = list(reader)

preprocessed_news_2 = []
for news in newslist_2:
    pre_news = dict()
    pre_news['text'] = text_preprocessing(news['text'])
    pre_news['domain'] = news['domain']
    pre_news['label'] = news['label']
    preprocessed_news_2.append(pre_news)
```    
## Xuất các kết quả tiền xử lý ra file CSV
```
preproc_path = './PreprocessingData'
```
### Xuất kết quả tiền xử lý vn_news_226_tlfr.csv ra preproc_vn_news_226_tlfr.csv
Shuffle thứ tự các tin tức trong preprocessed_news_1, tham khảo tại: [Shuffling a list of objects](https://stackoverflow.com/questions/976882/shuffling-a-list-of-objects/976918#976918).

```python
shuffle_id_list = [i for i in range(len(preprocessed_news_1))]
random.shuffle(shuffle_id_list)
```
Đưa kết quả preprocessing ra file preproc_vn_news_226_tlfr.csv. Tham khảo thêm tại: [How do I write a Python dictionary to a csv file? [duplicate]](https://stackoverflow.com/questions/10373247/how-do-i-write-a-python-dictionary-to-a-csv-file/10373268#10373268)

```python
keys1 = preprocessed_news_1[0].keys()
print(keys1)
print(type(preprocessed_news_1[0]))
dict_keys(['text', 'label'])
<class 'dict'>

with open(os.path.join(preproc_path, 'preproc_vn_news_226_tlfr.csv'), 'w') as csv_file:
    dict_writer = csv.DictWriter(csv_file, fieldnames = keys1)
    dict_writer.writeheader()
    for i in range(len(preprocessed_news_1)):
        dict_writer.writerow(preprocessed_news_1[shuffle_id_list[i]])
```
### Xuất kết quả tiền xử lý vn_news_223_tdlfr.csv ra preproc_vn_news_223_tdlfr.csv

```python
keys2 = preprocessed_news_2[0].keys()
print(keys2)
print(type(preprocessed_news_2[0]))
dict_keys(['text', 'domain', 'label'])
<class 'dict'>


with open(os.path.join(preproc_path, 'preproc_vn_news_223_tdlfr.csv'), 'w') as csv_file:
    dict_writer = csv.DictWriter(csv_file, fieldnames = keys2)
    dict_writer.writeheader()
    for news in preprocessed_news_2:
        dict_writer.writerow(news)
```

### Thử nghiệm

```
text = "ngày 21/11/2018, Đây là game GARRY'S MOD: tuyệt vô âm tín ==> => chơi trên kênh DR.Trực Tiếp Game test word: tu nhân tích đức  tructiepgame@gmail.com.vn của tôi - Dũng CT. tp.HCM, TP.HCM Rất mong nhận được sự ủng hộ của ae. ++ Đăng ký theo dõi kênh tại đây: https://goo.gl/A7BCZV ++ LINK TẢI APP CUBETV: https://www.cubetv.sg ++ THEO DÕI TTG TRÊN CUBETV: http://www.cubetv.sg/18947372 ++ DONATE ĐỂ TTG MUA ĐƯỢC NHIỀU GAME HƠN TẠI: - https://playerduo.com/tructiepgame - https://streamlabs.com
['ngày', '21/11/2018', ',', 'Đây', 'là', 'game', 'GARRY', "'", 'S', 'MOD', ':', 'tuyệt', 'vô', 'âm', 'tín', '==>', '=>', 'chơi', 'trên', 'kênh', 'DR.', 'Trực', 'Tiếp', 'Game', 'test', 'word', ':', 'tu', 'nhân', 'tích', 'đức', 'tructiepgame@gmail.com.vn', 'của', 'tôi', '-', 'Dũng', 'CT.', 'tp.', 'HCM', ',', 'TP.', 'HCM', 'Rất', 'mong', 'nhận', 'được', 'sự', 'ủng', 'hộ', 'của', 'ae.', '+', '+', 'Đăng', 'ký', 'theo', 'dõi', 'kênh', 'tại', 'đây', ':', 'https', ':', '/', '/', 'goo.', 'gl', '/', 'A7BCZV', '+', '+', 'LINK', 'TẢI', 'APP', 'CUBETV', ':', 'https', ':', '/', '/', 'www.', 'cubetv.', 'sg', '+', '+', 'THEO', 'DÕI', 'TTG', 'TRÊN', 'CUBETV', ':', 'http', ':', '/', '/', 'www.', 'cubetv.', 'sg', '/', '18947372', '+', '+', 'DONATE', 'ĐỂ', 'TTG', 'MUA', 'ĐƯỢC', 'NHIỀU', 'GAME', 'HƠN', 'TẠI', ':', '-', 'https', ':', '/', '/', 'playerduo.', 'com', '/', 'tructiepgame', '-', 'https', ':', '/', '/', 'streamlabs.', 'com', '/', 'tructiepgamevn', '+', '+', 'MUA', '/', 'THUÊ', 'GAME', 'BẢN', 'QUYỀN', 'GIÁ', 'RẺ', 'TẠI', ':', 'Website', ':', 'http', ':', '/', '/', 'divineshop.', 'vn', 'Fanpage', ':', 'https', ':', '/', '/', 'www.', 'facebook.', 'com', '/', 'Divine.', 'Shop.', '...', ',', 'da', 'che', 'mắt', 'ngựa']


token_list_1 = LongestMatching(token_list, bi_gram, tri_gram, four_gram)
print(token_list_1)
['ngày', '21/11/2018', ',', 'Đây', 'là', 'game', 'GARRY', "'", 'S', 'MOD', ':', 'tuyệt_vô_âm_tín', '==>', '=>', 'chơi', 'trên', 'kênh', 'DR.', 'Trực_Tiếp', 'Game', 'test', 'word', ':', 'tu_nhân_tích_đức', 'tructiepgame@gmail.com.vn', 'của', 'tôi', '-', 'Dũng', 'CT.', 'tp.', 'HCM', ',', 'TP.', 'HCM', 'Rất', 'mong', 'nhận', 'được', 'sự', 'ủng_hộ', 'của', 'ae.', '+', '+', 'Đăng_ký', 'theo_dõi', 'kênh', 'tại', 'đây', ':', 'https', ':', '/', '/', 'goo.', 'gl', '/', 'A7BCZV', '+', '+', 'LINK', 'TẢI', 'APP', 'CUBETV', ':', 'https', ':', '/', '/', 'www.', 'cubetv.', 'sg', '+', '+', 'THEO_DÕI', 'TTG', 'TRÊN', 'CUBETV', ':', 'http', ':', '/', '/', 'www.', 'cubetv.', 'sg', '/', '18947372', '+', '+', 'DONATE', 'ĐỂ', 'TTG', 'MUA', 'ĐƯỢC', 'NHIỀU', 'GAME', 'HƠN', 'TẠI', ':', '-', 'https', ':', '/', '/', 'playerduo.', 'com', '/', 'tructiepgame', '-', 'https', ':', '/', '/', 'streamlabs.', 'com', '/', 'tructiepgamevn', '+', '+', 'MUA', '/', 'THUÊ', 'GAME', 'BẢN_QUYỀN', 'GIÁ', 'RẺ', 'TẠI', ':', 'Website', ':', 'http', ':', '/', '/', 'divineshop.', 'vn', 'Fanpage', ':', 'https', ':', '/', '/', 'www.', 'facebook.', 'com', '/', 'Divine.', 'Shop.', '...', ',', 'da_che_mắt_ngựa']


preprocessing_text = text_preprocessing(text)
print(preprocessing_text)
ngày 21/11/2018 , Đây game GARRY ' S MOD : tuyệt_vô_âm_tín ==> => chơi kênh DR. Trực_Tiếp Game test word : tu_nhân_tích_đức tructiepgame@gmail.com.vn tôi - Dũng CT. tp. HCM , TP. HCM Rất mong nhận ủng_hộ ae. + + Đăng_ký theo_dõi kênh : https : / / goo. gl / A7BCZV + + LINK TẢI APP CUBETV : https : / / www. cubetv. sg + + THEO_DÕI TTG TRÊN CUBETV : http : / / www. cubetv. sg / 18947372 + + DONATE ĐỂ TTG MUA ĐƯỢC NHIỀU GAME HƠN TẠI : - https : / / playerduo. com / tructiepgame - https : / / streamlabs. com / tructiepgamevn + + MUA / THUÊ GAME BẢN_QUYỀN GIÁ RẺ TẠI : Website : http : / / divineshop. vn Fanpage : https : / / www. facebook. com / Divine. Shop. ... , da_che_mắt_ngựa
```          