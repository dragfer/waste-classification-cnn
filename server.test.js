const test = require("node:test");
const assert = require("node:assert/strict");
const http = require("node:http");
const { createServer, classes } = require("./server");

function request(server, requestPath) {
  return new Promise((resolve, reject) => {
    const address = server.address();
    const requestUrl = `http://127.0.0.1:${address.port}${requestPath}`;
    http.get(requestUrl, (response) => {
      let body = "";
      response.setEncoding("utf8");
      response.on("data", (chunk) => { body += chunk; });
      response.on("end", () => resolve({ status: response.statusCode, body }));
    }).on("error", reject);
  });
}

test("health endpoint exposes the four supported classes", async (t) => {
  const server = createServer().listen(0);
  t.after(() => server.close());
  const response = await request(server, "/api/health");
  assert.equal(response.status, 200);
  assert.deepEqual(JSON.parse(response.body).classes, classes);
});

test("root serves the frontend", async (t) => {
  const server = createServer().listen(0);
  t.after(() => server.close());
  const response = await request(server, "/");
  assert.equal(response.status, 200);
  assert.match(response.body, /WasteLens|waste materials/i);
});
