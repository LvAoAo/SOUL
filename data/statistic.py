
import joblib
import nltk

from nltk.corpus import brown, gutenberg, webtext, nps_chat, reuters, inaugural

# 下载语料库
# nltk.download('brown')
# nltk.download('gutenberg')
# nltk.download('webtext')
# nltk.download('nps_chat')
# nltk.download('reuters')
# nltk.download('inaugural')


# 加载语料库
corpus = nltk.corpus.brown.raw()

# 将所有字母转为小写
corpus = corpus.lower()

# 过滤非字母字符
corpus = ''.join(filter(str.isalpha, corpus))

# 统计每个字母的出现次数
freq_dist = nltk.FreqDist(corpus)
sum = 0
for freq in freq_dist.values():
    sum += freq
for char, freq in freq_dist.items():
    freq_dist[char] = freq / sum
# 打印每个字母的出现次数
joblib.dump(freq_dist, './char.pkl')
# x = joblib.load('char_freq.pkl')

fds = [nltk.FreqDist(brown.words()),
               nltk.FreqDist(gutenberg.words()),
               nltk.FreqDist(nps_chat.words()),
               nltk.FreqDist(webtext.words()),
               nltk.FreqDist(reuters.words()),
               nltk.FreqDist(inaugural.words()),
               ]
sum = 0
word_freq = {}
for fd in fds:
    for word, freq in fd.items():
        if word.lower() not in word_freq.keys():
            word_freq[word.lower()] = 0
        word_freq[word.lower()] += freq
        sum += freq
for word, freq in word_freq.items():
    print(freq)
    word_freq[word] = freq / sum
joblib.dump(word_freq, './word.pkl')
