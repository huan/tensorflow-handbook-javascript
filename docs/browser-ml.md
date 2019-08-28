
## 在浏览器中运行TensorFlow

![Chrome Machine Learning](images/chrome-ml.png)

TensorFlow.js可以让我们直接在浏览器中加载TensorFlow，让用户立即通过本地的CPU/GPU资源进行我们所需要的机器学习运算，更灵活的进行AI应用的开发。

浏览器中进行机器学习，相对比与服务器端来讲，将拥有以下四大优势：

1. 不行要安装软件或驱动（打开浏览器即可使用）；
1. 可以通过浏览器进行更加方便的人机交互；
1. 可以通过手机浏览器，调用手机硬件的各种传感器（如：GPS、电子罗盘、加速度传感器、摄像头等）；
1. 用户的数据可以无需上传到服务器，在本地即可完成所需操作。

通过这些优势，TensorFlow.js将带给开发者带来极高的灵活性。比如在 Google Creative Lab 在2018年7月发布的 Move Mirror 里，我们可以在手机上打开浏览器，通过手机摄像头检测视频中用户的身体动作姿势，然后通过对图片数据库中类似身体动作姿势的检索，给用户显示一个最能够和他当前动作相似的照片。在Move Mirror的运行过程中，数据没有上传到服务器，所有的运算都是在手机本地，基于手机的CPU/GPU完成的，而这项技术，将使Servreless与AI应用结合起来成为可能。

![Move Mirror](images/move-mirror.jpg)

- Move Mirror 地址：<https://experiments.withgoogle.com/move-mirror>
- Move Mirror 所使用的 PoseNet 地址：<https://github.com/tensorflow/tfjs-models/tree/master/posenet>

### 一个浏览器中的例子

线性回归的浏览器代码
