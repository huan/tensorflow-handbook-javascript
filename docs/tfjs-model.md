## 使用 Tensorflow.js 模型库

Tensorflow.js 提供了一系列预训练好的模型，方便大家快速的给自己的程序引入人工智能能力。

模型库 GitHub 地址：<https://github.com/tensorflow/tfjs-models>，其中模型分类包括：

1. 图像识别
1. 语音识别
1. 人体姿态识别
1. 物体识别
1. 文字分类

由于这些API默认模型文件都存储在谷歌云上，直接使用会导致中国用户无法直接读取。在程序内使用模型API时要提供 modelUrl 的参数，可以指向谷歌中国的镜像服务器。

谷歌云的base url是 <https://storage.googleapis.com>， 中国镜像的base url是 <https://www.gstaticcnapps.cn> 模型的url path是一致的。以 posenet模型为例：

- 谷歌云地址是：**https://storage.googleapis.com**/tfjs-models/savedmodel/posenet/mobilenet/float/050/model-stride16.json
- 中国镜像地址是：**https://www.gstaticcnapps.cn**/tfjs-models/savedmodel/posenet/mobilenet/float/050/model-stride16.json
