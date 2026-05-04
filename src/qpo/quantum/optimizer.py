"""Quantum optimization layer using QAOA on Lightning.qubit / Lightning.gpu.

AC-1.4: Pipeline contract integrity
The interface between the classical pipeline and the quantum layer is explicit:
QUBO matrix in, shortlist of variant IDs out. Swapping backends requires zero changes
outside this module.

Phase 1 QUBO encoding: diagonal-only. H_C = -Σ_i score_i * Z_i. The cost Hamiltonian
drives qubits toward |1⟩ for high-scoring candidates. Phase 2 will add off-diagonal
correlation terms. The penalty constraint (Σ x_i = K) is omitted in Phase 1; shortlisting
is done by taking the K qubits with highest marginal probability of |1⟩.
"""

import logging
import random
import warnings
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)


class QuantumOptimizer:
    """Quantum optimization via QAOA on Lightning.qubit (CPU) or Lightning.gpu (CUDA)."""

    def __init__(
        self,
        backend: str = "lightning",
        circuit_depth: int = 1,
        num_iterations: int = 30,
        seed: int = 42,
    ) -> None:
        self.backend = backend
        self.circuit_depth = circuit_depth
        self.num_iterations = num_iterations
        self.seed = seed

    def qubo_to_shortlist(
        self,
        qubo_matrix: np.ndarray,
        candidate_ids: Sequence[str],
        shortlist_size: int = 10,
    ) -> list[str]:
        """Convert QUBO matrix to shortlist via QAOA or stub.

        Args:
            qubo_matrix: NxN matrix; diagonal encodes pre-scores
            candidate_ids: Variant IDs corresponding to rows/cols
            shortlist_size: Number of top candidates to return

        Returns:
            List of variant IDs ranked by QAOA (best first)
        """
        n = len(candidate_ids)
        if qubo_matrix.shape[0] != n:
            raise ValueError(
                f"QUBO matrix size {qubo_matrix.shape[0]} != candidate_ids length {n}"
            )

        if self.backend == "stub":
            return random.sample(list(candidate_ids), min(shortlist_size, n))

        if self.backend in ("lightning", "lightning-qubit", "lightning-gpu"):
            return self._qaoa_lightning(qubo_matrix, list(candidate_ids), shortlist_size)

        return random.sample(list(candidate_ids), min(shortlist_size, n))

    def _qaoa_lightning(
        self,
        qubo_matrix: np.ndarray,
        candidate_ids: list[str],
        shortlist_size: int,
    ) -> list[str]:
        import pennylane as qml

        n_qubits = len(candidate_ids)
        scores = np.diag(qubo_matrix).tolist()

        dev = self._select_device(n_qubits)
        logger.info("QAOA: %d qubits, depth=%d, device=%s", n_qubits, self.circuit_depth, dev)

        @qml.qnode(dev)
        def cost_circuit(params: np.ndarray) -> float:
            _apply_qaoa_layers(params, scores, n_qubits, self.circuit_depth)
            # Cost Hamiltonian expectation: Σ_i -score_i * <Z_i>
            H = qml.Hamiltonian(
                [-scores[i] for i in range(n_qubits)],
                [qml.PauliZ(i) for i in range(n_qubits)],
            )
            return qml.expval(H)

        @qml.qnode(dev)
        def marginal_circuit(params: np.ndarray) -> list:
            _apply_qaoa_layers(params, scores, n_qubits, self.circuit_depth)
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        rng = np.random.default_rng(self.seed)
        init_params = rng.uniform(0, 2 * np.pi, 2 * self.circuit_depth)

        iteration_count = [0]

        def callback(params: np.ndarray) -> None:
            cost = float(cost_circuit(params))
            logger.debug("QAOA iteration %d: cost=%.6f", iteration_count[0], cost)
            iteration_count[0] += 1

        result = minimize(
            cost_circuit,
            init_params,
            method="COBYLA",
            options={"maxiter": self.num_iterations},
            callback=callback,
        )
        logger.info(
            "QAOA converged in %d iterations, final cost=%.6f",
            iteration_count[0],
            float(result.fun),
        )

        # p(qubit_i = |1⟩) = (1 - <Z_i>) / 2
        z_exps = marginal_circuit(result.x)
        qubit_probs = [(1.0 - float(z)) / 2.0 for z in z_exps]

        # Return top-K candidates by marginal selection probability
        ranked = sorted(range(n_qubits), key=lambda i: -qubit_probs[i])
        return [candidate_ids[i] for i in ranked[:shortlist_size]]

    def _select_device(self, n_qubits: int):
        import pennylane as qml

        if self.backend == "lightning-qubit":
            return qml.device("lightning.qubit", wires=n_qubits)

        try:
            dev = qml.device("lightning.gpu", wires=n_qubits)
            logger.info("GPU acceleration active: lightning.gpu (CUDA)")
            return dev
        except Exception as exc:
            warnings.warn(
                f"lightning.gpu unavailable ({exc}), falling back to lightning.qubit. "
                "Install CUDA support: pip install pennylane-lightning[gpu]",
                RuntimeWarning,
                stacklevel=3,
            )
            return qml.device("lightning.qubit", wires=n_qubits)


def _apply_qaoa_layers(
    params: np.ndarray,
    scores: list[float],
    n_qubits: int,
    circuit_depth: int,
) -> None:
    """Apply QAOA circuit layers in-place on the active PennyLane device.

    Defined at module level so both cost and marginal QNodes share identical
    circuit structure (required for correct parameter gradient flow).
    """
    import pennylane as qml

    # Uniform superposition
    for i in range(n_qubits):
        qml.Hadamard(wires=i)

    # p alternating cost + mixer layers
    for layer in range(circuit_depth):
        gamma = params[2 * layer]
        beta = params[2 * layer + 1]

        # Cost unitary: exp(-i γ H_C) where H_C = -Σ score_i Z_i
        # RZ(θ)|ψ⟩ = exp(-i θ/2 Z)|ψ⟩  →  θ = 2γ score_i
        for i in range(n_qubits):
            qml.RZ(2.0 * gamma * scores[i], wires=i)

        # Mixer unitary: exp(-i β H_M) where H_M = Σ X_i
        # RX(θ)|ψ⟩ = exp(-i θ/2 X)|ψ⟩  →  θ = 2β
        for i in range(n_qubits):
            qml.RX(2.0 * beta, wires=i)
