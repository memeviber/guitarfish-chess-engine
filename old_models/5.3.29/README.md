# Neural Network (5.3.29@6.4)

| Component | Specification |
| :--- | :--- |
| **Input Features** | 1,729 sparse geometric features (up to 256 active per position) |
| **Feature Transformer** | Dual accumulator ($2 \times 1024 \rightarrow 2048$), separate White/Black perspectives |
| **Hidden Layers** | Linear($2048 \rightarrow 1024$) $\rightarrow$ Linear($1024 \rightarrow 256$) $\rightarrow$ Linear($256 \rightarrow 64$) |
| **Activation** | SCReLU ($\text{clamp}(x, 0, 1)^2$) |
| **Output** | Linear($64 \rightarrow 1$) mapped to centipawns via WDL scaling |
| **Training Dataset** | 17.6M deduplicated, quiet positions |
| **Loss Function** | Dual Loss: $0.5 \times \text{BCE}(\text{WDL}) + 0.5 \times \text{Huber}(\text{Centipawns})$ |
