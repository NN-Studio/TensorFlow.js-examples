// 导入 TensorFlow.js 库
import tf from "@tensorflow/tfjs";
// 导入 TensorFlow.js 可视化库
import tfvis from "@tensorflow/tfjs-vis";

// 导入 IRIS 数据集和类别名称
import { getTrisData, IRIS_CLASSES } from "./data.js";

// 加载数据集，按 85:15 比例分割训练集和测试集
const [xTrain, yTrain, xTest, yTest] = getTrisData(0.15);

// console.log(xTrain, yTrain, xTest, yTest);
// x格式: 4个特征 [花萼长度, 花萼宽度, 花瓣长度, 花瓣宽度]
// Tensor
//     [[4.8000002, 3.0999999, 1.6      , 0.2      ],
//      ...,
//      [5.8000002, 2.7      , 5.0999999, 1.9      ]]
// y格式: 独热编码 [setosa, versicolor, virginica]
// Tensor
//     [[1, 0, 0],
//      ...,
//      [0, 0, 1]]
// xTrain.print();
// yTrain.print();
// xTest.print();
// yTest.print();

// 创建顺序模型
const model = tf.sequential();

// 隐藏层：10 个神经元，输入为 4 个特征
model.add(tf.layers.dense({
    units: 10,
    inputShape: [4],
    activation: "sigmoid"
}));

// 输出层：3 个神经元对应 3 个鸢尾花类别
model.add(tf.layers.dense({
    units: 3,
    activation: "softmax"  // 输出概率分布
}));

model.compile({

    // 设置交叉熵损失函数
    // (对数损失函数的多分类版本)
    loss: "categoricalCrossentropy",

    optimizer: tf.train.adam(0.1),

    // 设置准确度
    metrics: ["accuracy"]
});

// 训练模型
model.fit(xTrain, yTrain, {
    epochs: 100,  // 训练轮数
    validationData: [xTest, yTest],  // 验证集
    callbacks: tfvis.show.fitCallbacks(
        { name: "训练效果" },
        ["loss", "val_loss", "acc", "val_acc"],  // 监控指标
        { callbacks: ["onEpochEnd"] }  // 每轮结束后更新图表
    )
}).then(function () {
    // 训练完成后进行预测
    const pred = model.predict(tf.tensor([[5.1, 3.5, 1.4, 0.2]]));
    let result = pred.dataSync();

    // 输出各类别的概率
    console.log([
        IRIS_CLASSES[0] + (result[0] * 100).toFixed(2) + "%",
        IRIS_CLASSES[1] + (result[1] * 100).toFixed(2) + "%",
        IRIS_CLASSES[2] + (result[2] * 100).toFixed(2) + "%"
    ].join("\n"));
});