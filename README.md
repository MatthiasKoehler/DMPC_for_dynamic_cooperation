# Distributed Model Predictive Control for Dynamic Cooperation of Multi-Agent Systems

![Python](https://img.shields.io/badge/Python-3.13-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
[![DOI](https://img.shields.io/badge/DOI-10.48550/arXiv.2504.00225-blue)](https://doi.org/10.48550/arXiv.2504.00225)


This repository contains the simulation code used for all numerical examples in

> M. Köhler, M. A. Müller, and F. Allgöwer, 'Distributed Model Predictive Control for Dynamic Cooperation of Multi-Agent Systems,' 2025, arXiv:2504.00225. doi: [10.48550/arXiv.2504.00225](https://doi.org/10.48550/arXiv.2504.00225)

In addition, the data shown are provided for further evaluation and animations illustrating the examples are included.

*The code in this repository was developed solely by the first author. The other authors of the associated publication were not involved in its implementation or maintenance and bear no responsibility for its correctness or completeness.*

## Usage

The code was run using *Python 3.13.1*, with the required packages listed in `requirements.txt`.

To run the code:

1. Install Python 3.13.1 (or a compatible version).
2. Create a virtual environment, activate it, and install the dependencies using (on Windows):

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Notes on implementation

Please note that this is conceptual code, implemented without consideration for modularity, flexibility, or computational efficiency.

This repository includes a custom implementation of the decentralised SQP scheme from:

> G. Stomberg, A. Engelmann, M. Diehl, and T. Faulwasser, 'Decentralized real-time iterations for distributed nonlinear model predictive control,' 2024, arXiv:2401.14898. doi: [10.48550/arXiv.2401.14898](https://doi.org/10.48550/arXiv.2401.14898)

For an efficient implementation, refer to: [dmpc_rto](https://github.com/optcon/dmpc_rto)

Also see:

> G. Stomberg, H. Ebel, T. Faulwasser, and P. Eberhard, 'Cooperative distributed MPC via decentralized real-time optimization: Implementation results for robot formations,' *Control Eng. Pract.*, vol. 138, 105579, 2023. doi: [10.1016/j.conengprac.2023.105579](https://doi.org/10.1016/j.conengprac.2023.105579)

Terminal constraints for the quadrotor example are computed following:

> J. Köhler, M. A. Müller, and F. Allgöwer, 'A nonlinear model predictive control framework using reference generic terminal ingredients,' *IEEE Trans. Autom. Control*, vol. 65 (3), 3576--3583, 2019. doi: [10.1109/TAC.2019.2949350](https://doi.org/10.1109/TAC.2019.2949350)

This implementation uses [CasADi](https://web.casadi.org/docs/) and [CVXPY](https://www.cvxpy.org/).

## Citation

If you use this code in your research, please cite the following:

```bash
@article{MKoehler2025,
  title={Distributed Model Predictive Control for Dynamic Cooperation of Multi-Agent Systems},
  author={K{\"o}hler, Matthias and M{\"u}ller, Matthias A. and Allg{\"o}wer}, Frank},
  year={2025},
  journal={arxiv:2504.00225},
  doi = {10.48550/arXiv.2504.00225},
}
```

Please also cite the relevant works and dependancies listed above if you use the decentralised SQP scheme or the terminal constraint design.
