# Third-Party Notices

This repository includes components derived from the open-source projects
listed below. Each component keeps the license under which it was received;
nothing in this file or in the repository's own LICENSE alters those terms.

The corresponding LICENSE file is also placed in each subdirectory of
`third_party/`.

---

## 1. NIID-Bench

**Used in:** `third_party/niid_bench/partition.py`

The Dirichlet label-distribution-skew partitioner is derived from the
`partition_data` function (`noniid-labeldir` strategy) of NIID-Bench.

- Project: https://github.com/Xtra-Computing/NIID-Bench
- Reference: Q. Li, Y. Diao, Q. Chen and B. He, "Federated Learning on
  Non-IID Data Silos: An Experimental Study," ICDE 2022.
- License: MIT License
- Copyright: Copyright (c) 2021 Yiqun Diao, Qinbin Li

Modifications by Fujitsu Limited are confined to `s2wef/partition.py`,
which wraps this component without altering its sampling procedure.

```
MIT License

Copyright (c) 2021 Yiqun Diao, Qinbin Li

Copyright (c) 2020 International Business Machines

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```

---

## 2. pytorch-cifar

**Used in:** `third_party/pytorch_cifar/resnet.py`

The CIFAR-oriented ResNet definition (`BasicBlock` and `ResNet`) is derived
from `models/resnet.py` of kuangliu/pytorch-cifar.

- Project: https://github.com/kuangliu/pytorch-cifar
- Reference: K. He, X. Zhang, S. Ren and J. Sun, "Deep Residual Learning
  for Image Recognition," CVPR 2016.
- License: MIT License
- Copyright: 2017 liukuang

```
MIT License

Copyright (c) 2017 liukuang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 3. Runtime dependencies

The following packages are required at runtime but are not redistributed as
part of this repository. Their licenses are listed for reference; verify each
against the internal license summary before publication.

| Package | License |
| --- | --- |
| PyTorch (`torch`) | BSD 3-Clause |
| torchvision | BSD 3-Clause |
| NumPy | BSD 3-Clause |
| SciPy | BSD 3-Clause |
| scikit-learn | BSD 3-Clause |
| pandas | BSD 3-Clause |
| Matplotlib | Matplotlib License (BSD-compatible) |

The datasets themselves (MNIST, Adult, CIFAR-10, CIFAR-100) are downloaded
by the user at run time and are not redistributed here.