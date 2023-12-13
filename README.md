# BERT with tract

An example of running BERT QA with tract.

The code here is mostly ported from the iOS example of Bert QA in the tensorflow lite repo: https://github.com/tensorflow/examples/tree/master/lite/examples/bert_qa/ios

Download the mobile bert model in tflite format via:

```
curl -SLo mobilebert.tflite https://tfhub.dev/tensorflow/lite-model/mobilebert/1/metadata/1?lite-format=tflite
```

Then the tflite model needs to be converted to ONNX:

```
python -m tf2onnx.convert --opset 16 --tflite mobilebert.tflite --output mobilebert.onnx
```

Then this example can be ran with `cargo run --release`.
