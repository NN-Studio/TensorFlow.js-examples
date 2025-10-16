import tf from "@tensorflow/tfjs";
import tfvis from "@tensorflow/tfjs-vis";

const xs = [1, 2, 3, 4];
const ys = [1, 3, 5, 7];

tfvis.render.scatterplot({
    name: "线性回归训练集"
}, {
    values: xs.map((x, i) => ({ x, y: ys[i] }))
}, {
    xAxisDomain: [0, 5],
    yAxisDomain: [0, 8]
});

// 创建一个顺序模型
const model = tf.sequential();

// 添加一个全链接层
model.add(tf.layers.dense({
    units: 1, // 神经元个数
    inputShape: [1], // 输入形状
}));

model.compile({

    // 设置损失函数
    // 这里使用均方误差
    loss: tf.losses.meanSquaredError,

    // 设置优化器
    // 0.1表示学习率
    optimizer: tf.train.sgd(0.1)

});

const inputs = tf.tensor(xs);
const labels = tf.tensor(ys);

// 训练
model.fit(inputs, labels, {
    batchSize: 2, // 每次，也就是一个小批量用几个数据
    epochs: 100, // 迭代多少次

    // 显示训练的过程
    callbacks: tfvis.show.fitCallbacks(
        { name: "训练过程" },

        // 我们将确定要监控的指标
        ["loss"])

}).then(function () {

    // 用模型预测
    // 这里x=5的时候，看看预测y是多少
    const output = model.predict(tf.tensor([5]));

    // output.print();
    console.log(output.dataSync()[0]);

});
