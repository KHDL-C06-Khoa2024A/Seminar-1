import numpy as np

def get_topn_keywords_model(text, model, topn):
    feature_array = np.array(model.get_params()['tfidfvect'].get_feature_names())
    response = model.get_params()['tfidfvect'].transform([text])
    tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]
    top_n = feature_array[tfidf_sorting][:topn]
    return top_n