# VFND Experiences - Thí nghiệm trên tập dữ liệu VFND

Một số thí nghiệm dựa trên dataset [VFND](https://github.com/thanhhocse96/vfnd-vietnamese-fake-news-datasets) sẽ được thực hiện trong repo này

## 1. Mô tả nội dung các file và thư mục

1. ```corpus_extending.ipynb```: dùng để mở rộng bộ từ điển để dùng cho phương pháp Longest Matching, bộ từ điển mở rộng được đặt trong thư mục ```Dictionaries```

2. ```news-preprocessing.ipynb```: thử nghiệm các phương thức tiền xử lý dữ liệu
3. Thư mục ```Dictionaries```: Chứa các bộ từ điển ```bi_gram.txt, tri_gram.txt, four_gram.txt``` và ```Stopwords_vi.txt```

## 2. Tham khảo và vay mượn
### 2.1 Vay mượn mã nguồn
1. [urlmarker.py](https://github.com/rcompton/ryancompton.net/blob/master/assets/praw_drugs/urlmarker.py): Mã nguồn hỗ trợ trích xuất tương đối chính xác URL trong văn bản, Tham khảo theo [\[1\]](http://ryancompton.net/2015/02/16/url-extraction-in-python/)
2. Các bộ từ điển tham khảo từ [VNLP Core](https://github.com/deepai-solutions/core_nlp), [Từ điển tiếng Việt](http://www.informatik.uni-leipzig.de/~duc/Dict/) và bộ ```Stopwords_vi.txt``` tham khảo tại [dnanhkhoa/Stopwords_vi.txt](https://gist.github.com/dnanhkhoa/724a20e5e8dce42d2dda99c601190dfc)

### 2.2 Tài liệu tham khảo
1. [Url extraction in python - Ryan Compton](http://ryancompton.net/2015/02/16/url-extraction-in-python/): Trích xuất URL trong văn bản bằng REGEX
2. [Các bài viết VNLP Core - Forum MachineLearning cơ bản](https://forum.machinelearningcoban.com/t/vnlp-core-1-bai-toan-tach-tu-tieng-viet-tokenization-word-segmentation/2002): [Bài 1](https://forum.machinelearningcoban.com/t/vnlp-core-1-bai-toan-tach-tu-tieng-viet-tokenization-word-segmentation/2002), [Bài 2](https://forum.machinelearningcoban.com/t/vnlp-core-2-thuc-hanh-training-va-su-dung-bieu-dien-tu-trong-khong-gian-word-embedding/2101) & [Bài 3](https://forum.machinelearningcoban.com/t/vnlp-core-3-bai-toan-phan-loai-van-ban-phan-tich-cam-xuc-cua-binh-luan-text-classification/2371)

