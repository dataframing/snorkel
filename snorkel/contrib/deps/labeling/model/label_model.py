from itertools import chain, product
from typing import Any, List, Optional, Tuple

import numpy as np

from snorkel.labeling.model.graph_utils import get_clique_tree
from snorkel.labeling.model.label_model import CliqueData, LabelModel


class DependencyAwareLabelModel(LabelModel):
    """A LabelModel that handles dependencies and learn associated weights to assign training labels.

    The model is based on the matrix factorization approach detailed here: https://arxiv.org/pdf/1810.02840.pdf,
    and uses Robust PCA as an intermediate step as shown here: https://arxiv.org/pdf/1903.05844.pdf.
    """

    def _get_augmented_label_matrix(self, L: np.ndarray) -> np.ndarray:
        """Create augmented version of label matrix.

        In augmented version, each column is an indicator
        for whether a certain source or clique of sources voted in a certain
        pattern.

        Parameters
        ----------
        L
            An [n,m] label matrix with values in {0,1,...,k}

        Returns
        -------
        np.ndarray
            An [n,m*k] dense matrix with values in {0,1}
        """
        L_ind = super()._get_augmented_label_matrix(L)

        # Get the higher-order clique statistics based on the clique tree
        # First, iterate over the maximal cliques (nodes of c_tree) and
        # separator sets (edges of c_tree)
        if self.higher_order:
            L_aug = np.copy(L_ind)
            for item in chain(self.c_tree.nodes(), self.c_tree.edges()):
                if isinstance(item, int):
                    C = self.c_tree.node[item]
                    C_type = "node"
                elif isinstance(item, tuple):
                    C = self.c_tree[item[0]][item[1]]
                    C_type = "edge"
                else:
                    raise ValueError(item)
                members = list(C["members"])

                nc = len(members)

                # If a unary maximal clique, just store its existing index
                if nc == 1:
                    C["start_index"] = members[0] * self.cardinality
                    C["end_index"] = (members[0] + 1) * self.cardinality

                # Else add one column for each possible value
                else:
                    L_C = np.ones((self.n, self.cardinality ** nc))
                    for i, vals in enumerate(
                        product(range(self.cardinality), repeat=nc)
                    ):
                        for j, v in enumerate(vals):
                            L_C[:, i] *= L_ind[:, members[j] * self.cardinality + v]

                    # Add to L_aug and store the indices
                    if L_aug is not None:
                        C["start_index"] = L_aug.shape[1]
                        C["end_index"] = L_aug.shape[1] + L_C.shape[1]
                        L_aug = np.hstack([L_aug, L_C])
                    else:
                        C["start_index"] = 0
                        C["end_index"] = L_C.shape[1]
                        L_aug = L_C

                    # Add to self.c_data as well
                    id = tuple(members) if len(members) > 1 else members[0]
                    self.c_data[id] = CliqueData(
                        start_index=C["start_index"],
                        end_index=C["end_index"],
                        max_cliques=set([item]) if C_type == "node" else set(item),
                    )
            return L_aug
        else:
            return L_ind

    def _set_structure(self) -> None:
        nodes = range(self.m)
        self.c_tree = get_clique_tree(nodes, self.deps)
        if len(self.deps) > 0:
            self.higher_order = True
        else:
            self.higher_order = False

    def fit_with_deps(
        self,
        L_train: np.ndarray,
        Y_dev: Optional[np.ndarray] = None,
        learn_deps: Optional[bool] = True,
        thresh_mult: Optional[float] = 0.5,
        gamma: Optional[float] = 1e-8,
        lam: Optional[float] = 0.1,
        deps: Optional[List[Tuple[int, int]]] = None,
        class_balance: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> None:
        """Train label model using dependencies (if given).

        Train label model to estimate mu, the parameters used to combine LFs.

        Parameters
        ----------
        L_train
            An [n,m] matrix with values in {-1,0,1,...,k-1}
        Y_dev
            Gold labels for dev set for estimating class_balance, by default None
        learn_deps
            Whether to learn dependencies, by default True
        thresh_mult
            Threshold multiplier for selecting thresh_mult * max off diagonal entry from sparse matrix
        gamma
            Parameter in objective function related to sparsity
        lam
            Parameter in objective function related to sparsity and low rank
        deps
            Optional list of pairs of correlated LF indices.
        class_balance
            Each class's percentage of the population, by default None
        **kwargs
            Arguments for changing train config defaults

        Raises
        ------
        Exception
            If loss in NaN

        Examples
        --------
        >>> L = np.array([[0, 0, -1], [-1, 0, 1], [1, -1, 0]])
        >>> Y_dev = [0, 1, 0]
        >>> label_model = DependencyAwareLabelModel(verbose=False)
        >>> label_model.fit_with_deps(L, deps=[(0, 2)])  # doctest: +SKIP
        >>> label_model.fit_with_deps(L, deps=[(0, 2)], Y_dev=Y_dev)  # doctest: +SKIP
        >>> label_model.fit_with_deps(L, deps=[(0, 2)], class_balance=[0.7, 0.3])  # doctest: +SKIP
        >>> label_model.fit_with_deps(L, learn_deps=True)  # doctest: +SKIP
        """
        self.deps = deps
        self.fit(L_train, Y_dev, class_balance, **kwargs)
