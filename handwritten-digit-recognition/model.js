import tf from "@tensorflow/tfjs";

/**
 * 创建手写数字识别模型 (MNIST)
 * 使用卷积神经网络 (CNN) 架构
 * @returns {tf.Sequential} 返回训练好的模型
 */
export default function () {
    // 创建一个顺序模型（层按顺序堆叠）
    const model = tf.sequential();

    // MNIST 数据集图片规格：28x28 像素的灰度图
    const IMAGE_WIDTH = 28;
    const IMAGE_HEIGHT = 28;
    const IMAGE_CHANNELS = 1;

    // 输入层
    model.add(tf.layers.conv2d({

        // 将流入模型第一层的数据的形状。
        // 在本例中，我们的 MNIST 示例是 28x28 像素的黑白图片。
        // 图片数据的规范格式为 [row, column, depth]，因此在这里我们需要配置以下形状：[28, 28, 1]。
        // 各个维度的像素数量为 28 行和 28 列，深度为 1，因为我们的图片只有一个颜色通道。
        // 请注意，我们不会在输入形状中指定批次大小。
        // 层设计为与批次大小无关，因此在推理期间，您可以传入任何批次大小的张量。
        inputShape: [IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS],

        // 要应用于输入数据的滑动卷积过滤器窗口的尺寸
        kernelSize: 5,

        // 要应用于输入数据的尺寸为 kernelSize 的过滤器窗口数量
        filters: 8,

        // 滑动窗口的“步长”，即每次移动图片时过滤器都会移动多少像素
        strides: 1,

        // 卷积完成后应用于数据的激活函数
        // 此处用的是修正线性单元 (ReLU) 函数
        activation: 'relu',

        // 随机初始化模型权重的方法
        kernelInitializer: 'varianceScaling'
    }));

    // 最大池化层：降低特征图尺寸，减少计算量和过拟合
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    // 第二个卷积层：提取更高级的特征
    model.add(tf.layers.conv2d({
        kernelSize: 5,
        filters: 16,  // 增加过滤器数量，捕捉更多特征
        strides: 1,
        activation: 'relu',
        kernelInitializer: 'varianceScaling'
    }));

    // 第二个最大池化层
    model.add(tf.layers.maxPooling2d({ poolSize: [2, 2], strides: [2, 2] }));

    // 展平数据表示法
    // 图片是高维数据，而卷积运算往往会增大传入其中的数据的大小。
    // 在将数据传递到最终分类层之前，我们需要将数据展平为一个长数组。
    // 密集层（我们会用作最终层）只需要采用 tensor1d，因而此步骤在许多分类任务中很常见。
    model.add(tf.layers.flatten());

    // 输出层：全连接层，输出 10 个数字的概率分布
    model.add(tf.layers.dense({
        units: 10,  // 10 个输出对应 0-9 十个数字
        kernelInitializer: 'varianceScaling',
        activation: 'softmax'  // 输出概率分布，总和为 1
    }));

    // 编译模型：配置优化器、损失函数和评估指标
    model.compile({
        optimizer: tf.train.adam(),  // Adam 优化器，自适应学习率
        loss: 'categoricalCrossentropy',  // 多分类交叉熵损失
        metrics: ['accuracy'],  // 监控准确率
    });

    return model;
}