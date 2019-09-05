## 在服务器端使用 TensorFlow.js

服务器端使用 JavaScript ，首先需要按照 [NodeJS.org](https://nodejs.org) 官网的说明，完成安装最新版本的 Node.js 。

然后，完成以下四个步骤即可完成配置：

1. 确认 Node.js 版本

TensorFlow.js 需要 Node.js 版本建议使用 10 或更高版本。

```shell
$ node --verion
v10.5.0

$ npm --version
6.4.1
```

2. 建立 TensorFlow.js 项目目录

```shell
$ mkdir tfjs
$ cd tfjs
```

3. 安装 TensorFlow.js

```shell
# 初始化项目管理文件 package.json
$ npm init -y

# 安装 tfjs 库，纯 JavaScript 版本
$ npm install @tensorflow/tfjs 

# 安装 tfjs-node 库，C Binding 版本
$ npm install @tensorflow/tfjs-node 

# 安装 tfjs-node-gpu 库，支持 CUDA GPU 加速
$ npm install @tensorflow/tfjs-node-gpu
```

4. 确认 Node.js 和 TensorFlow.js 工作正常

```shell
$ node
> require('@tensorflow/tfjs').version
{ 'tfjs-core': '1.0.1',
  'tfjs-data': '1.0.1',
  'tfjs-layers': '1.0.1',
  'tfjs-converter': '1.0.1',
  tfjs: '1.0.1' }
> 
```

如果你看到了上面的 `tfjs-core`, `tfjs-data`, `tfjs-layers` 和 `tfjs-converter` 的输出信息，那么就说明环境配置没有问题了。
