# Model adaptation in a discrete fracture network: existence of solutions and numerical strategies

Source code and examples for the paper<br>
"*Model adaptation in a discrete fracture network: existence of solutions and numerical strategies*" by Alessio Fumagalli, Francesco Saverio Patacchini. See [arXiv pre-print](https://arxiv.org/abs/XXX).


# Reproduce results from paper
Runscripts for all test cases of the work available [here](./examples).<br>
Note that you may have to revert to an older version of [PorePy](https://github.com/pmgbergen/porepy) to run the examples.

# Abstract
Fractures are normally present in the underground and are, for some physical
processes, of paramount importance. Their accurate description is 
fundamental to obtain reliable numerical outcomes useful, e.g., for energy
management. Depending on the physical and geometrical properties of the
fractures, fluid flow can behave differently, going from a slow Darcian
regime to more complicated Brinkman or even Forchheimer regimes for high velocity.
The main problem is to determine where in the fractures one regime is more adequate
than others. In order to determine these low-speed and high-speed regions, this 
work proposes an adaptive strategy which is based on selecting the appropriate 
constitutive law linking velocity and pressure according to a threshold criterion
on the magnitude of the fluid velocity itself. Both theoretical and numerical 
aspects are considered and investigated, showing the potentiality of the 
proposed approach. From the analytical viewpoint, we show existence of
weak solutions to such model under reasonable hypotheses on the constitutive laws.
To this end, we use a variational approach identifying solutions with minimizers of
an underlying energy functional. From the numerical viewpoint, we propose a
one-dimensional algorithm which tracks the interface between the low- and high-speed
regions. By running numerical experiments using this algorithm, we illustrate 
some interesting behaviors of our adaptive model on a single fracture and small
networks of intersecting fractures.

# Citing
If you use this work in your research, we ask you to cite the following publication [arXiv: XX [math.NA]](https://arxiv.org/abs/XX).

# PorePy version
If you want to run the code you need to install [PorePy](https://github.com/pmgbergen/porepy) and revert to commit 26857d19a5e77a880c245495867428a945326b19 <br>
Newer versions of PorePy may not be compatible with this repository.

# License
See [license](./LICENSE).
