import Canvas from "vislite/lib/Canvas/index.es.js";
import { setStyle } from "oipage/web/style/index.js";
import tf from "@tensorflow/tfjs";

export default function (callback) {
    let el = document.createElement("div");

    setStyle(el, {
        "width": "400px",
        "height": "100vh",
        "position": "fixed",
        "left": 0,
        "top": 0,
        // "background-color": "rgb(0 0 0 / 16%)",
        "z-index": "100000"
    });

    let contentEl = document.createElement("div");
    setStyle(contentEl, {
        "width": "280px",
        "position": "absolute",
        "left": "50%",
        "top": "50%",
        "background-color": "white",
        "transform": "translateX(-50%) translateY(-50%)",
        "text-align": "center",
        "user-select": "none",
        "font-size": "0"
    });
    el.appendChild(contentEl);

    let canvasEl = document.createElement("div");
    contentEl.appendChild(canvasEl);

    setStyle(canvasEl, {
        "background-color": "rgb(0 0 0 / 16%)"
    });

    let painter = new Canvas(canvasEl, {}, 280, 280).config({
        lineWidth: 30,
        lineCap: "round"
    });

    let isStart = false;

    painter.bind("mousedown", function (_, x, y) {
        isStart = true;
        painter.beginPath().moveTo(x, y);
    });

    painter.bind("mousemove", function (_, x, y) {
        if (isStart) {
            painter.lineTo(x, y).stroke().moveTo(x, y);
        }
    });

    painter.bind("mouseup", function (_, x, y) {
        isStart = false;
        painter.lineTo(x, y).stroke();
    });

    let btnsEl = document.createElement("div");
    setStyle(btnsEl, {
        "line-height": "44px",
        "height": "44px",
        "font-size": "16px",
        "background-color": "#bfa670",
        "color": "#1c1c1c"
    });
    contentEl.appendChild(btnsEl);

    let submitEl = document.createElement("div");
    submitEl.innerText = "识别";
    btnsEl.appendChild(submitEl);
    submitEl.addEventListener("click", function () {

        let data = [];
        for (let i = 0; i < 28; i++) {
            for (let j = 0; j < 28; j++) {
                if (!data[j]) data[j] = [];
                data[j][i] = [painter.getColor(i * 10 + 5, j * 10 + 5) === "rgba(0,0,0,255)" ? 1 : 0];
                // data[j][i] = [painter.getColor(i * 10 + 5, j * 10 + 5).replace("rgba(", "").replace(")", "").split(",")[3] / 255];
            }
        }

        callback(tf.tensor([data], [1, 28, 28, 1]));
        painter.clearRect(0, 0, 280, 280);
    });

    let cancelEl = document.createElement("div");
    cancelEl.innerText = "清空";
    btnsEl.appendChild(cancelEl);
    cancelEl.addEventListener("click", function () {
        painter.clearRect(0, 0, 280, 280);
    });

    for (let btnEl of [submitEl, cancelEl]) {
        setStyle(btnEl, {
            "display": "inline-block",
            "cursor": "pointer",
            "padding": "0 30px"
        });
    }

    document.body.appendChild(el);
}