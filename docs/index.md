> **Atwood’s Law**
>  
> “Any application that can be written in JavaScript, will eventually be written in JavaScript.”
>  
>  -- Jeff Atwood, Founder of StackOverflow.com

> “JavaScript now works.”
>  
>  -- Paul Graham, YC Founder

# JavaScript 版 Tensorflow

![Tensorflow.js](images/tensorflow-js.gif)

Tensorflow.js 是 Tensorflow 的 JavaScript 版本，支持GPU硬件加速，可以运行在 Node.js 或浏览器环境中。它不但支持完全基于 JavaScript 从头开发、训练和部署模型，也可以用来运行已有的 Python 版 Tensorflow 模型，或者基于现有的模型进行继续训练。

![Tensorflow.js Architecture](images/architecture.gif)

Tensorflow.js 支持 GPU 硬件加速。在 Node.js 环境中，如果有 CUDA 环境支持，或者在浏览器环境中，有 WebGL 环境支持，那么 Tensorflow.js 可以使用硬件进行加速。

本章，我们将基于 Tensorflow.js 1.0，向大家简单的介绍如何基于 ES6 的 JavaScript 进行 Tensorflow.js 的开发，然后提供两个例子，并基于例子进行详细的讲解和介绍，最终实现使用纯 JavaScript 进行 Tensorflow 模型的开发、训练和部署。

1. [在浏览器中使用 Tensorflow.js](browser-ml.md)
1. [基本的回归模型](regression-model.md)
1. [在服务器端使用 Tensorflow.js](nodejs-ml.md)
1. [Seq2Seq 闲聊对话模型](seq2seq-model.md)
1. [将 Python 模型转换为 Tensorflow.js 可以加载的版本](converter-js.md)
1. [Tensorflow.js 性能对比](performance.md)


## 本章 GitHub 代码仓库

本章中提到的 JavaScript 版 Tensorflow 的相关代码，使用说明，和训练好的模型文件及参数，都可以在作者的 GitHub 上找到。地址： https://github.com/huan/javascript-concise-chitchat
