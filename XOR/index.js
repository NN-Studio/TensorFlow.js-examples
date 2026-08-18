import tf from "@tensorflow/tfjs";
import tfvis from "@tensorflow/tfjs-vis";

// 导入 XOR 数据集
import { getData } from "./data.js";

// 获取 400 个训练样本
const data = getData(400);

// 可视化训练数据，按标签分组显示
tfvis.render.scatterplot({
    name: "XOR训练数据"
}, {
    values: [
        data.filter(p => p.label === 1),
        data.filter(p => p.label === 0)
    ]
});

// 创建顺序模型
const model = tf.sequential();

// 添加隐藏层：4个神经元，ReLU 激活函数（解决非线性问题）
model.add(tf.layers.dense({
    units: 4,
    inputShape: [2],  // 2个输入特征
    activation: "relu"  // 非线性激活函数
}));

// 添加输出层：1个神经元，输出 0-1 概率
model.add(tf.layers.dense({
    units: 1,
    activation: "sigmoid"  // Sigmoid 激活函数
}));

// 编译模型
model.compile({
    loss: tf.losses.logLoss,  // 对数损失函数
    optimizer: tf.train.adam(0.1)  // Adam 优化器
});

// 将数据转换为张量
const inputs = tf.tensor(data.map(p => [p.x, p.y]));
const labels = tf.tensor(data.map(p => p.label));

// 训练模型
model.fit(inputs, labels, {
    epochs: 10,  // 训练轮数
    callbacks: tfvis.show.fitCallbacks(
        { name: "训练过程" },
        ["loss"])  // 监控损失值
}).then(function () {
    // 训练完成，进行预测
    const pred = model.predict(tf.tensor([[7, 7]]));  // 预测点 (7, 7) 的类别概率
    console.log(pred.dataSync()[0]);  // 输出预测概率
});