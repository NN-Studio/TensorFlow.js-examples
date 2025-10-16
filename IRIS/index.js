import tf from "@tensorflow/tfjs";
import tfvis from "@tensorflow/tfjs-vis";

import { getTrisData, IRIS_CLASSES } from "./data.js";

const [xTrain, yTrain, xTest, yTest] = getTrisData(0.15);
// console.log(xTrain, yTrain, xTest, yTest);
// x格式
// Tensor
//     [[4.8000002, 3.0999999, 1.6      , 0.2      ],
//      ...,
//      [5.8000002, 2.7      , 5.0999999, 1.9      ]]
// y格式
// Tensor
//     [[1, 0, 0],
//      ...,
//      [0, 0, 1]]
// xTrain.print();
// yTrain.print();
// xTest.print();
// yTest.print();

const model = tf.sequential();

model.add(tf.layers.dense({
    units: 10,
    inputShape: [4],
    activation: "sigmoid"
}));

model.add(tf.layers.dense({
    units: 3,
    activation: "softmax"
}));

model.compile({

    // 设置交叉熵损失函数
    // (对数损失函数的多分类版本)
    loss: "categoricalCrossentropy",

    optimizer: tf.train.adam(0.1),

    // 设置准确度
    metrics: ["accuracy"]
});

model.fit(xTrain, yTrain, {
    epochs: 100,

    //  设置验证集
    validationData: [xTest, yTest],

    callbacks: tfvis.show.fitCallbacks(
        { name: "训练效果" },

        // 训练集的损失
        // 验证集的损失
        // 训练集的准确度
        // 验证集的准确度
        ["loss", "val_loss", "acc", "val_acc"],
        {
            // 不默认显示onBatchEnd了
            callbacks: ["onEpochEnd"]
        }
    )
}).then(function () {

    const pred = model.predict(tf.tensor([[5.1, 3.5, 1.4, 0.2]]));

    let result = pred.dataSync();
    console.log([
        IRIS_CLASSES[0] + (result[0] * 100).toFixed(2) + "%",
        IRIS_CLASSES[1] + (result[1] * 100).toFixed(2) + "%",
        IRIS_CLASSES[2] + (result[2] * 100).toFixed(2) + "%"
    ].join("\n"));

});