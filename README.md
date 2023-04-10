## FPN-Capsnet

Combination of Feature Pyramid Network (FPN) & Capsule Network (Capsnet), applied on MNIST dataset.


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

Install dependent libraries

* pip
  ```sh
  pip install requirements.txt
  ```

### To Train and Test

1. Run ```main.py``` file with below arguments

| Argument      | Options |
| ----------- | ----------- |
| --model | ['fpn-capsnet', 'ipyramid-capsnet'] |
| --device | ['cuda', 'cpu'] |
| --criterion | ['capsule-loss', 'cross-entropy'] |
| --lr_scheduler | ['exponential', 'step', 'none'] |
| --lr_decay | \<any floats\> |
| --epochs | \<any integers\> |
| --lr | \<any floats\> |
| --batch_size | \<any integers\> |

2. The Trained model & log will be added at "./src/model/"
3. The Test results will be added at"./src/output/"

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACCURACY EXAMPLES -->
## Accuracy

Accuracy on Kaggle

![Accuracy](./src/image/kaggle.png)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- REFERENCES -->
## References

* [https://arxiv.org/pdf/1710.09829.pdf]
* [https://arxiv.org/pdf/1612.03144.pdf]
* [https://arxiv.org/pdf/1512.03385.pdf]

<p align="right">(<a href="#readme-top">back to top</a>)</p>
