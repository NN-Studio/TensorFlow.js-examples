const oipagePluginCommon = require("./oipage-plugin-common");

module.exports = {
    devServer: {
        port: 20000,
        baseUrl: "./",
        cache: true,
        intercept: [{
            test: /@tensorflow\/tfjs$/,
            handler: oipagePluginCommon("@tensorflow/tfjs", "./node_modules/@tensorflow/tfjs/dist/tf.min.js")
        }, {
            test: /@tensorflow\/tfjs-vis$/,
            handler: oipagePluginCommon("@tensorflow/tfjs-vis", "./node_modules/@tensorflow/tfjs-vis/dist/tfjs-vis.umd.min.js")
        }]
    }
};