import tf from "@tensorflow/tfjs";
import tfvis from "@tensorflow/tfjs-vis";

import { getData } from "./data.js";

const data = getData(400);
// console.log(data);

tfvis.render.scatterplot({
    name: "逻辑回归训练数据"
}, {
    values: [
        data.filter(p => p.label === 1),
        data.filter(p => p.label === 0)
    ]
});

const model = tf.sequential();

model.add(tf.layers.dense({
    units: 1,
    inputShape: [2],

    // 设置激活函数
    activation: "sigmoid"
}));

model.compile({

    // 设置损失函数
    // 这里使用对数损失函数
    loss: tf.losses.logLoss,

    // 设置优化器
    // adam可以自己调学习率
    optimizer: tf.train.adam(0.1)
});

const inputs = tf.tensor(data.map(p => [p.x, p.y]));
const labels = tf.tensor(data.map(p => p.label));

model.fit(inputs, labels, {
    batchSize: 40,
    epochs: 50,
    callbacks: tfvis.show.fitCallbacks(
        { name: "训练过程" },
        ["loss"])
}).then(function () {

    const pred = model.predict(tf.tensor([[7, 7]]));
    console.log(pred.dataSync()[0]);

});