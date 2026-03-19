import inspect
import importlib
import unittest
from unittest.mock import patch

import torch

swaa = importlib.import_module("swaa_patch.hack_hf_swaa")


class LinearMemOpsInterfaceTest(unittest.TestCase):
    def test_linear_mem_ops_signature_accepts_fla_state_kwargs(self) -> None:
        params = inspect.signature(swaa.linear_mem_ops).parameters
        self.assertIn("initial_state", params)
        self.assertIn("output_final_state", params)

    def test_linear_mem_ops_transposes_to_bthd_and_forwards_state(self) -> None:
        captured: dict[str, object] = {}
        initial_state = torch.randn(2, 4, 7, 11)

        def fake_fla(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            initial_state: torch.Tensor | None = None,
            output_final_state: bool | None = None,
            **kwargs,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:
            captured["q_shape"] = tuple(q.shape)
            captured["k_shape"] = tuple(k.shape)
            captured["v_shape"] = tuple(v.shape)
            captured["initial_state"] = initial_state
            captured["output_final_state"] = output_final_state

            out = torch.zeros(
                q.shape[0],
                q.shape[1],
                q.shape[2],
                v.shape[-1],
                dtype=v.dtype,
                device=v.device,
            )
            return out, initial_state

        q = torch.randn(2, 4, 5, 7)
        k = torch.randn(2, 2, 5, 7)
        v = torch.randn(2, 2, 5, 11)

        with patch.object(swaa, "fused_recurrent_linear_attn", side_effect=fake_fla):
            out, final_state = swaa.linear_mem_ops(
                object(),
                q=q,
                k=k,
                v=v,
                initial_state=initial_state,
                output_final_state=True,
                mode="fused_recurrent",
            )

        self.assertEqual(captured["q_shape"], (2, 5, 4, 7))
        self.assertEqual(captured["k_shape"], (2, 5, 4, 7))
        self.assertEqual(captured["v_shape"], (2, 5, 4, 11))
        self.assertIs(captured["initial_state"], initial_state)
        self.assertTrue(captured["output_final_state"])
        self.assertEqual(tuple(out.shape), (2, 5, 44))
        self.assertIs(final_state, initial_state)


if __name__ == "__main__":
    unittest.main()
