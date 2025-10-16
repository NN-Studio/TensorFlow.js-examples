import tf from "@tensorflow/tfjs";
import tfvis from "@tensorflow/tfjs-vis";

import { getData } from "./data.js";

const data = getData(400);
// console.log(data);

tfvis.render.scatterplot({
    name: "XOR训练数据"
}, {
    values: [
        data.filter(p => p.label === 1),
        data.filter(p => p.label === 0)
    ]
});

const model = tf.sequential();

// 添加一个隐藏层
model.add(tf.layers.dense({
    units: 4,

    // 只有第一层需要设置输入形状
    inputShape: [2],

    // 设置激活函数
    // 用非线性的激活函数即可
    activation: "relu"
}));

// 添加输出层
model.add(tf.layers.dense({
    units: 1,

    // 设置激活函数
    // 输出的是一个0-1的概率
    activation: "sigmoid"
}));

model.compile({
    loss: tf.losses.logLoss,
    optimizer: tf.train.adam(0.1)
});

const inputs = tf.tensor(data.map(p => [p.x, p.y]));
const labels = tf.tensor(data.map(p => p.label));

model.fit(inputs, labels, {
    epochs: 10,
    callbacks: tfvis.show.fitCallbacks(
        { name: "训练过程" },
        ["loss"])
}).then(function () {

    const pred = model.predict(tf.tensor([[7, 7]]));
    console.log(pred.dataSync()[0]);

});