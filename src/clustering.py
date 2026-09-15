import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# NOTE: umap is imported lazily inside __init__ rather than at module scope.
# umap/__init__.py imports parametric_umap unconditionally, so ANY import of
# umap loads TensorFlow when TensorFlow is installed -- and TensorFlow
# segfaults (SIGSEGV) when it shares a process with PyTorch, which the TCN
# risk overlay needs. Deferring the import keeps `import src.clustering`
# free of TensorFlow; use compute_clusters_isolated() below to get cluster
# labels in a process that also has to use PyTorch.


class FactorClusterer:

    def __init__(
        self,
        n_components: int = 2,
        eps: float = 0.4,
        min_samples: int = 2,
        n_neighbors: int = 5,
        min_dist: float = 0.05,
        metric: str = "cosine",
        n_epochs: int = 200,
        random_state: int = 42,
        parametric: bool = True,
    ):
        
        """Initializes Parametric UMAP non-linear manifold reduction coupled with DBSCAN.

        :param n_components: Target latent space dimensions (2 or 3 for
        optimal DBSCAN density).
        :param eps: Maximum radius for DBSCAN neighborhood evaluation in UMAP
        space.
        :param min_samples: Minimum asset count required to establish a core
        cluster.
        :param n_neighbors: Local metric neighborhood size for UMAP manifold
        construction.
        :param min_dist: Effective minimum distance between embedded points in
        latent space.
        :param metric: Distance metric for high-dimensional loadings (cosine
        aligns directional betas).
        :param n_epochs: Number of training epochs for the Parametric UMAP
        neural network.
        :param parametric: Use the neural ParametricUMAP encoder, which
        learns a reusable mapping so out-of-sample loadings can be
        projected via transform(). Set False for plain UMAP, which fits
        faster but cannot embed new points. Neither avoids loading
        TensorFlow -- see the note at the top of this file.
        """
        
        self.n_components = n_components
        self.eps = eps
        self.min_samples = min_samples

        self.scaler = StandardScaler()
        self.parametric = parametric

        common = dict(
            n_components=self.n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            n_epochs=n_epochs,
            random_state=random_state,
            verbose=False,
        )
        if parametric:
            # Imported here, not at module scope, so that `parametric=False`
            # never loads TensorFlow. See the note at the top of this file.
            from umap.parametric_umap import ParametricUMAP

            self.umap_reducer = ParametricUMAP(**common)
        else:
            from umap import UMAP

            self.umap_reducer = UMAP(**common)

        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels_ = None
        self.clustered_assets_ = None
        self.umap_embeddings_ = None

    def fit(self, factor_loadings: pd.DataFrame) -> "FactorClusterer":
        
        """Fits Parametric UMAP on PCA factor loadings and clusters assets via DBSCAN."""
        
        # 1. Standardize factor loadings
        norm_loadings = self.scaler.fit_transform(factor_loadings.values)

        # 2. Train neural network encoder and project to latent topological space
        self.umap_embeddings_ = self.umap_reducer.fit_transform(norm_loadings)

        # 3. Fit DBSCAN on low-dimensional, high-density UMAP coordinates
        self.dbscan.fit(self.umap_embeddings_)
        self.labels_ = self.dbscan.labels_

        # 4. Map cluster labels to tickers
        self.clustered_assets_ = pd.DataFrame(
            {
                "ticker": factor_loadings.index,
                "cluster": self.labels_,
                "umap_dim1": self.umap_embeddings_[:, 0],
                "umap_dim2": self.umap_embeddings_[:, 1],
            }
        ).set_index("ticker")

        return self

    def transform(self, new_factor_loadings: pd.DataFrame) -> np.ndarray:
        
        """Projects out-of-sample or rolling-window factor loadings into the learned

        latent space using the trained neural encoder.
        """
        
        norm_new = self.scaler.transform(new_factor_loadings.values)
        return self.umap_reducer.transform(norm_new)

    def get_clusters(self) -> dict[int, list[str]]:
        
        """Returns dictionary of cluster assignments, where -1 denotes unclustered noise."""
        
        if self.clustered_assets_ is None:
            raise ValueError(
                "Model has not been fitted. Call fit(factor_loadings) first."
            )

        clusters = {}
        for cluster_id in np.unique(self.labels_):
            tickers = self.clustered_assets_[
                self.clustered_assets_["cluster"] == cluster_id
            ].index.tolist()
            clusters[cluster_id] = tickers
        
        return clusters


def compute_clusters_isolated(
    factor_loadings: pd.DataFrame,
    n_components: int = 2,
    eps: float = 0.4,
    min_samples: int = 2,
    parametric: bool = True,
) -> dict[int, list[str]]:
    """Clusters factor loadings in a separate process and returns the labels.

    UMAP drags in TensorFlow, which segfaults if PyTorch is also active in
    the same process. The TCN risk overlay needs PyTorch, so it cannot call
    FactorClusterer directly. Running the clustering in a short-lived
    subprocess keeps TensorFlow out of the caller entirely: the child exits
    before the caller touches torch, and only the label mapping crosses back.

    :param factor_loadings: PCA factor loadings (N assets x K components).
    :return: ``{cluster_id: [tickers]}``, with -1 denoting DBSCAN noise.
    :raises RuntimeError: If the subprocess fails, with its stderr attached.
    """
    import json
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "loadings.parquet"
        dst = Path(tmp) / "clusters.json"
        factor_loadings.to_parquet(src)

        script = (
            "import json, pandas as pd\n"
            "from src.clustering import FactorClusterer\n"
            f"L = pd.read_parquet({str(src)!r})\n"
            f"c = FactorClusterer(n_components={n_components}, eps={eps}, "
            f"min_samples={min_samples}, parametric={parametric}).fit(L)\n"
            "out = {str(k): v for k, v in c.get_clusters().items()}\n"
            f"json.dump(out, open({str(dst)!r}, 'w'))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        if proc.returncode != 0 or not dst.exists():
            raise RuntimeError(
                f"Isolated clustering failed (exit {proc.returncode}).\n"
                f"{proc.stderr[-2000:]}"
            )
        return {int(k): v for k, v in json.load(open(dst)).items()}
