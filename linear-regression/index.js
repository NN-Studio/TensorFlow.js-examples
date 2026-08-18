import tf from "@tensorflow/tfjs";
import tfvis from "@tensorflow/tfjs-vis";

// 训练数据：y = 2x - 1
const xs = [1, 2, 3, 4];
const ys = [1, 3, 5, 7];

// 可视化训练数据集
tfvis.render.scatterplot({
    name: "线性回归训练集"
}, {
    values: xs.map((x, i) => ({ x, y: ys[i] }))
}, {
    xAxisDomain: [0, 5],
    yAxisDomain: [0, 8]
});

// 创建顺序模型
const model = tf.sequential();

// 添加全连接层：单输入单输出的线性层
model.add(tf.layers.dense({
    units: 1,  // 输出维度
    inputShape: [1],  // 输入维度
}));

// 编译模型
model.compile({
    loss: tf.losses.meanSquaredError,  // 均方误差损失函数
    optimizer: tf.train.sgd(0.1)  // 随机梯度下降优化器，学习率0.1
});

// 将数据转换为张量
const inputs = tf.tensor(xs);
const labels = tf.tensor(ys);

// 训练模型
model.fit(inputs, labels, {
    batchSize: 2,  // 批次大小，也就是一个小批量用几个数据
    epochs: 100,  // 训练轮数（迭代多少次）

    // 显示训练的过程
    callbacks: tfvis.show.fitCallbacks(
        { name: "训练过程" },

        // 我们将确定要监控的指标
        ["loss"]  // 监控损失值
    )
}).then(function () {
    // 训练完成，用训练好的模型进行预测
    // 这里x=5的时候，看看预测y是多少
    const output = model.predict(tf.tensor([5]));  // 预测 x=5 时的 y 值

    // output.print();
    console.log(output.dataSync()[0]);  // 输出预测结果
});
