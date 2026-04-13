async function predict() {

    const session = await ort.InferenceSession.create(
        "titanic_model.onnx",
        {
            executionProviders: ['wasm'],
            enableMemPattern: false,
            executionMode: 'sequential'
        }
    );

    const input = new Float32Array([
        parseFloat(f1.value),
        parseFloat(f2.value),
        parseFloat(f3.value),
        parseFloat(f4.value),
        parseFloat(f5.value),
        parseFloat(f6.value),
        parseFloat(f7.value)
    ]);

    const tensor = new ort.Tensor("float32", input, [1,7]);

    const feeds = {};
    feeds[session.inputNames[0]] = tensor;

    const results = await session.run(feeds);

    const output = Object.values(results)[0].data;

    const prediction = output[0] > output[1] ? "Did NOT Survive" : "Survived";

    document.getElementById("result").innerText =
    "Prediction: " + prediction +
    "\nConfidence: " + Math.max(...output).toFixed(3);
}