# Text Matching Chinese
基于Pytorch实现多个中文文本匹配模型，使用的数据是来自哈工大提供的中文问题文本匹配数据[Corpus](http://icrc.hitsz.edu.cn/info/1037/1146.htm)，如需使用该数据，请自行发邮件去申请使用。

## 一、训练词向量

使用中文维基百科语料训练word2vec词向量，具体的训练过程请见：[基于中文维基百科文本数据训练词向量](https://github.com/ChiYeungLaw/WordEmbedding-WikiChinese)。

## 二、各种模型对比

运行环境：

- Python 3.6
- Pytorch 1.2
- GTX 1080ti

利用模型在测试集上的准确率作为评价指标。

结果比较：

| 模型 | ACC  |
| :--: | :--: |
| ESIM |      |

## 三、参考

- [(ESIM)](https://arxiv.org/abs/1609.06038) Enhanced LSTM for Natural Language Inference

