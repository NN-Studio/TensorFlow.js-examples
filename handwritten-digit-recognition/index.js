import tf from "@tensorflow/tfjs";
import tfvis from "@tensorflow/tfjs-vis";

import { MnistData } from "./mnist_data.js";
import getModel from "./model.js";
import showInput from "./showInput.js";
import showExamples from "./showExamples.js";
import { showAccuracy, showConfusion } from "./showEvaluation.js";

let data = new MnistData();
data.load().then(function () {
    showExamples(data);

    // 定义模型
    let model = getModel();
    // tfvis.show.modelSummary({ name: 'Model Architecture', tab: 'Model' }, model);

    // 用于训练模型的训练集
    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(5500); // 从训练集返回随机批次的图片及其标签
        return [
            d.xs.reshape([5500, 28, 28, 1]),
            d.labels
        ];
    });

    // 测试模型的验证集
    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(1000); // 从测试集中返回一批图片及其标签
        return [
            d.xs.reshape([1000, 28, 28, 1]),
            d.labels
        ];
    });

    // console.log(trainXs,trainYs);
    // trainXs.print();
    // trainYs.print();

    // 训练模型
    model.fit(trainXs, trainYs, {
        batchSize: 512,
        validationData: [testXs, testYs],
        epochs: 10,
        shuffle: true,
        callbacks: tfvis.show.fitCallbacks(
            { name: "训练过程" },
            ['loss', 'val_loss', 'acc', 'val_acc'])
    }).then(function () {
        showAccuracy(model, data);
        showConfusion(model, data);

        showInput(function (inputX) {
            console.log(inputX)

            const imageTensor = tf.tidy(() => {
                // Reshape the image to 28x28 px
                return inputX.reshape([28, 28, 1]);
            });

            const surface =
                tfvis.visor().surface({ name: 'User Input Data', tab: 'Input Data' });
            const canvas = document.createElement('canvas');
            canvas.width = 28;
            canvas.height = 28;
            canvas.style = 'margin: 4px;';
            tf.browser.toPixels(imageTensor, canvas).then(function () {
                surface.drawArea.appendChild(canvas);
                imageTensor.dispose();
            });

            // console.log(inputX);
            inputX.print();
            const output = model.predict(inputX).dataSync();
            // console.log(output);
            let result = [];

            let value1 = 0, value2 = 0;
            for (let i = 0; i <= 9; i++) {
                result.push("是" + i + "的概率为：" + (output[i] * 100).toFixed(2) + "%");
                if (output[i] > value1) {
                    value1 = output[i];
                    value2 = i;
                }
            }
            alert(result.join("  ") + "\n\n结果为：" + value2);

        });

    });

});