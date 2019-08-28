## 用JavaScript加载Python版本存储格式的模型

一般TensorFlow的模型，以Python版本为例，会被存储为以下四种格式之一：

1. TensorFlow SavedModel
2. Frozen Model
3. TensorFlow Hub Module
4. Keras Module

Google 目前最佳实践中，推荐使用 SavedModel 方法进行模型保存。同时所有以上格式，都可以通过 tensorflowjs-converter 转换器，将其转换为可以直接被 TensorFlow.js 加载的格式，在JavaScript语言中进行使用。

### TensorFlow.js转换器tensorflowjs_converter

`tensorflowjs_converter`可以将Python存储的模型格式，转换为JavaScript可以直接调用的模型格式。

安装`tensorflowjs_converter`：

```shell
pip install tensorflowjs
```

`tensorflowjs_converter`的使用细节，可以通过`--help`参数查看程序帮助：

```shell
tensorflowjs_converter --help
```

以下我们以MobilenetV1为例，看一下如何对模型文件进行转换操作，并将可以被TensorFlow.js加载的模型文件，存放到`/mobilenet/tfjs_model`目录下。

### 转换SavedModel

将`/mobilenet/saved_model`转换到`/mobilenet/tfjs_model`：

```shell
tensorflowjs_converter \
    --input_format=tf_saved_model \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    --saved_model_tags=serve \
    /mobilenet/saved_model \
    /mobilenet/tfjs_model
```

### FrozenModel例子

将`/mobilenet/frozen_model.pb`转换到`/mobilenet/tfjs_model`：

```shell
tensorflowjs_converter \
    --input_format=tf_frozen_model \
    --output_node_names='MobilenetV1/Predictions/Reshape_1' \
    /mobilenet/frozen_model.pb \
    /mobilenet/tfjs_model
```

### Hub例子

将`https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/1`转换到`/mobilenet/tfjs_model`：

```shell
tensorflowjs_converter \
    --input_format=tf_hub \
    'https://tfhub.dev/google/imagenet/mobilenet_v1_100_224/classification/1' \
    /mobilenet/tfjs_model
```

### Keras例子

将`/tmp/model.h5`转换到`/tmp/tfjs_model`：

```shell
tensorflowjs_converter \
    --input_format keras \
    /tmp/model.h5 \
    /tmp/tfjs_model
```

## 用JavaScript加载和运行

为了加载转换完成的模型文件，我们需要安装`tfjs-converter`和`@tensorflow/tfjs`模块：

```shell
npm install @tensorflow/tfjs
```

然后，我们就可以通过JavaScript来加载TensorFlow模型了！

```js
import * as tf from '@tensorflow/tfjs';

const MODEL_URL = 'model_directory/model.json';

const model = await tf.loadGraphModel(MODEL_URL);
// 对Keras或者tfjs原生的层模型，使用下面的加载函数:
// const model = await tf.loadLayersModel(MODEL_URL);

const cat = document.getElementById('cat');
model.execute(tf.browser.fromPixels(cat))
```
