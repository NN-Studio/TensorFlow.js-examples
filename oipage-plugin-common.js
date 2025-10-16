const { readFileSync } = require("fs");
const { join } = require("path");

module.exports = function (name, url) {

    return function (request, response) {
        let head = {};
        head["Content-type"] = "application/javascript;charset=utf-8";
        head["ETag"] = name + "@v" + require(name + "/package.json").version;

        if (request.headers["if-none-match"] === head["ETag"]) {
            response.writeHead('304', head);
            response.end();
            console.log("<i> \x1b[1m\x1b[32m[OIPage-http-server] Cache File: " + name + "\x1b[0m " + new Date().toLocaleString() + "\x1b[33m\x1b[1m 304\x1b[0m");;
            return;
        }

        response.writeHead(200, head);

        let source = readFileSync(join(__dirname, url), {
            encoding: "utf8"
        });

        response.write(`let module = { exports: {}};
let exports = module.exports;
${source}
export default module.exports;
`);
        response.end();
    }
}