import tf from "@tensorflow/tfjs";
import tfvis from "@tensorflow/tfjs-vis";

// 训练数据：身高和体重
const heights = [150, 160, 170];
const weights = [40, 50, 60];

// 可视化训练数据
tfvis.render.scatterplot({
    name: "身高体重训练数据"
}, {
    values: heights.map((x, i) => ({ x, y: weights[i] }))
}, {
    xAxisDomain: [140, 180],
    yAxisDomain: [30, 70]
});

// 数据归一化：将数据映射到 [0, 1] 区间
// 身高: (x - 150) / 20，体重: (y - 40) / 20
const inputs = tf.tensor(heights).sub(150).div(20);
const labels = tf.tensor(weights).sub(40).div(20);

// 创建顺序模型
const model = tf.sequential();

// 添加全连接层
model.add(tf.layers.dense({
    units: 1,
    inputShape: [1],
}));

// 编译模型
model.compile({
    loss: tf.losses.meanSquaredError,  // 均方误差损失
    optimizer: tf.train.sgd(0.1)  // 随机梯度下降优化器
});

// 训练模型
model.fit(inputs, labels, {
    batchSize: 3,
    epochs: 100,
    callbacks: tfvis.show.fitCallbacks(
        { name: "训练过程" },
        ["loss"])
}).then(function () {
    // 预测前需归一化输入
    const output = model.predict(tf.tensor([180]).sub(150).div(20));
    // 结果需反归一化还原真实值
    console.log(output.mul(20).add(40).dataSync()[0]);
});
