# Browser model

The application expects a TensorFlow.js Graph model in this folder:

- `model.json`
- the shard files referenced by `model.json`

The model must accept a `[1, 224, 224, 3]` RGB tensor with values in `[0, 1]`
and return four scores in this exact order:

1. plastic
2. paper
3. glass
4. metal

The checked-in `model.json` and shard file are the browser export of the CNN
trained by this project's `training/train.py`. To replace it, train a new CNN
with your dataset, then export it with `tensorflowjs_converter` into this
directory.
The application reports that the model is unavailable rather than returning a
prediction when these files are absent or invalid.
