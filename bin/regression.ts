import * as tf from '@tensorflow/tfjs'

const xsRaw = tf.tensor([2013, 2014, 2015, 2016, 2017])
const ysRaw = tf.tensor([12000, 14000, 15000, 16500, 17500])

// 归一化
const xs = xsRaw.sub(xsRaw.min())
                .div(xsRaw.max().sub(xsRaw.min()))
const ys = ysRaw.sub(ysRaw.min())
                .div(ysRaw.max().sub(ysRaw.min()))

const a = tf.scalar(Math.random()).variable()
const b = tf.scalar(Math.random()).variable()

// y = a * x + b.
const f = (x: tf.Tensor) => a.mul(x).add(b)
const loss = (pred: tf.Tensor, label: tf.Tensor) => pred.sub(label).square().mean() as tf.Scalar

const learningRate = 1e-3
const optimizer = tf.train.sgd(learningRate)

// 训练模型
for (let i = 0; i < 10000; i++) {
   optimizer.minimize(() => loss(f(xs), ys))
}

// 预测
console.log(`a: ${a.dataSync()}, b: ${b.dataSync()}`)
const preds = f(xs).dataSync() as Float32Array
const trues = ys.arraySync() as number[]
preds.forEach((pred, i) => {
   console.log(`x: ${i}, pred: ${pred.toFixed(2)}, true: ${trues[i].toFixed(2)}`)
})
