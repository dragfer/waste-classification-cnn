const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
const { URL } = require("node:url");

const port = Number(process.env.PORT || 8080);
const webRoot = path.join(__dirname, "web");
const classes = ["plastic", "paper", "glass", "metal"];
const contentTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml"
};

function sendJson(response, statusCode, body) {
  response.writeHead(statusCode, { "content-type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(body));
}

function safeWebPath(requestPath) {
  const decoded = decodeURIComponent(requestPath);
  const relativePath = decoded === "/" ? "index.html" : decoded.slice(1);
  const filePath = path.resolve(webRoot, relativePath);
  return filePath.startsWith(webRoot + path.sep) ? filePath : null;
}

function createServer() {
  return http.createServer((request, response) => {
    const requestUrl = new URL(request.url || "/", "http://localhost");

    if (request.method !== "GET") {
      sendJson(response, 405, { error: "Only GET requests are supported." });
      return;
    }

    if (requestUrl.pathname === "/api/health") {
      const modelPath = path.join(webRoot, "model", "model.json");
      sendJson(response, 200, {
        status: "ok",
        classifier: "browser-cnn",
        model_loaded: fs.existsSync(modelPath),
        classes
      });
      return;
    }

    if (requestUrl.pathname === "/api/classes") {
      sendJson(response, 200, { classes });
      return;
    }

    const filePath = safeWebPath(requestUrl.pathname);
    if (!filePath) {
      sendJson(response, 400, { error: "Invalid path." });
      return;
    }

    fs.readFile(filePath, (error, data) => {
      if (error) {
        sendJson(response, error.code === "ENOENT" ? 404 : 500, {
          error: error.code === "ENOENT" ? "Not found." : "Unable to read resource."
        });
        return;
      }

      response.writeHead(200, {
        "content-type": contentTypes[path.extname(filePath).toLowerCase()] || "application/octet-stream",
        "cache-control": "no-cache"
      });
      response.end(data);
    });
  });
}

if (require.main === module) {
  createServer().listen(port, () => {
    console.log(`Waste classification application running at http://localhost:${port}`);
  });
}

module.exports = { createServer, classes };
