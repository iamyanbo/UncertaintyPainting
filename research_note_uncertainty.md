# Uncertainty Estimation Methods: Alternatives to Entropy

## 1. Monte Carlo (MC) Dropout (Bayesian Approximation)
*   **Concept:** Dropout is usually turned off during inference. If you keep it **on** and run the image through the network $N$ times (e.g., 10 times), you get 10 slightly different predictions.
*   **Calculation:**
    *   **Prediction:** Mean of the $N$ outputs.
    *   **Uncertainty:** Variance (standard deviation) of the $N$ outputs.
*   **Pros:** Works with **any** standard trained network (like DeepLab) without retraining. Captures "Epistemic" (model) uncertainty well.
*   **Cons:** Slower inference ($N$ times slower).

## 2. Evidential Deep Learning (Built-in Uncertainty)
*   **Concept:** The network is modified to output the parameters of a Dirichlet distribution (alphas) instead of just class probabilities. It learns to say "I don't know" by outputting a flat Dirichlet distribution.
*   **Calculation:** The network outputs `evidence` for each class. Uncertainty is inversely proportional to total evidence.
    *   $U = K / \sum(\alpha_k)$
*   **Pros:** **Single-pass** (fast). Explicitly separates "I haven't seen this data before" (Epistemic) from "This image is blurry" (Aleatoric).
*   **Cons:** Requires **retraining** the 2D network with a special loss function (EDL Loss). You cannot use a standard pre-trained DeepLab "out of the box" without fine-tuning.

## 3. Deep Ensembles
*   **Concept:** Train 5 different networks (random inits) and average their predictions.
*   **Calculation:** Variance across the 5 models.
*   **Pros:** Gold standard for accuracy and uncertainty quality.
*   **Cons:** Very expensive (train 5 models, run 5 models).

## 4. Learned Variance (Aleatoric Uncertainty)
*   **Concept:** The network has a second "head" that outputs a variance scalar $\sigma^2$ along with the class prediction.
*   **Calculation:** The model directly predicts its own error margin.
*   **Pros:** Fast single-pass.
*   **Cons:** Only captures "Aleatoric" uncertainty (noise), not "Epistemic" (model ignorance). Requires retraining.

## Conclusion

The choice of uncertainty method depends on the balance between computational cost and the type of uncertainty required (Aleatoric vs. Epistemic).

*   **Entropy:** Fastest baseline, captures total uncertainty but conflates data and model uncertainty.
*   **MC Dropout:** Captures epistemic uncertainty without retraining, at the cost of inference speed.
*   **Evidential Deep Learning:** Theoretially robust, separates uncertainty types, but requires custom training losses.

