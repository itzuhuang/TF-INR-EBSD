# TF-INR: Tensor-Factorized Implicit Neural Representation for EBSD Kikuchi Data

## Structure

* `config.py` is similar to a conventional configuration file. It contains the parameters that you will most likely need to change.

* `utils.py` contains the model architecture and helper functions.

* `01_train.py` is where you run the training. It generates:

  * The trained `.pth` model file
  * PCA explained variance plots (`.png` and `.svg`)
  * Learning curve plots (`.png` and `.svg`)

* `02_evaluate.py` generates:

  * MAE map
  * Comparisons (MAE in kikuchi level) between reconstructed and input Kikuchi patterns

This script also generates bandpass filtered `gt` and reconstructed Kikuchi patterns (with only five pattern pairs are shown for visualization. Filtering of the full map is applied only in `03_indexing.py`). This allows you to tune the filtering parameters before entering the indexing process. It also produces plots of the first 12 basis patterns.

* `03_indexing.py` generates the `gt` and reconstructed `.mat` files for further preferred analysis.

* `04_gradient_profile.py` generates the gradient profile plots (`.png` and `.svg`). For details, please see: [the cool paper]

## Note

In the code, the input patterns are called `gt(GT)` (`Ground Truth`), although they may not really be the absolute truth. Still, they are the only “truth” that the machine can see. Perhaps much of the information available to us is not truly the truth either, but it is the truth from which we construct our own world.

We use SIREN to reconstruct Kikuchi patterns, then take the gradient of the reconstruction and find that it looks remarkably like a grain boundary, which fundamentally makes sense.

But the author is wondering: if we could take the divergence of the gradient of our own life history, what would we get? It is kind of an interesting question.

If you run the code from `01` to `04`, you will find a story inspired by this video:

https://youtu.be/QZigkM0-6QE?si=RlFmnHv5NfJH4b3H

A dehydrated man is lost in a desert and asks the soul’s most direct question:

> “If there is nowhere to go, what is the point of walking?”

(This is also the question the author asks herself during every machine learning training attempt.)

The author acknowledges that there is no simple answer, so how about we make ourselves some rigatoni pasta?
