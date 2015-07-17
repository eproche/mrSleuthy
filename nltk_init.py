#!/usr/bin/env python

import nltk

def download_nltk_corpora():
    '''Download required nltk corpora with optional proxy information'''

    # Proxy settings: check the following website for information
    # http://www.nltk.org/data.html#installing-via-a-proxy-web-server
    # nltk.set_proxy(PROXY_URL, (PROXY_USERNAME, PROXY_PASSWORD))

    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('maxent_treebank_pos_tagger')

if __name__ == '__main__':
    download_nltk_corpora()