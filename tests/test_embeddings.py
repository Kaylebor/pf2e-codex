from __future__ import annotations

from pathlib import Path

import pytest

from pf2e_codex.embeddings import ONNXProvider, _pin_onnx_shapes


def test_pin_onnx_shapes_preserves_external_tensor_data(tmp_path: Path) -> None:
    onnx = pytest.importorskip("onnx")
    numpy_helper = pytest.importorskip("onnx.numpy_helper")
    np = pytest.importorskip("numpy")

    input_info = onnx.helper.make_tensor_value_info(
        "input_ids", onnx.TensorProto.FLOAT, ["batch_size", "sequence_length"]
    )
    output_info = onnx.helper.make_tensor_value_info(
        "output", onnx.TensorProto.FLOAT, ["batch_size", "sequence_length"]
    )
    weight = numpy_helper.from_array(np.ones((2, 2), dtype=np.float32), name="weight")
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["input_ids"], ["output"])],
        "external-data-test",
        [input_info],
        [output_info],
        [weight],
    )
    model = onnx.helper.make_model(graph)
    model_path = tmp_path / "model.onnx"
    weights_path = tmp_path / "model.onnx_data"
    onnx.save_model(
        model,
        model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=weights_path.name,
        size_threshold=0,
    )
    weight_bytes = weights_path.read_bytes()

    _pin_onnx_shapes(model_path, seq_len=512)

    pinned = onnx.load(model_path, load_external_data=False)
    sequence_dim = pinned.graph.input[0].type.tensor_type.shape.dim[1]
    assert sequence_dim.dim_value == 512
    assert not sequence_dim.dim_param
    assert pinned.graph.initializer[0].data_location == onnx.TensorProto.EXTERNAL
    external_data = {entry.key: entry.value for entry in pinned.graph.initializer[0].external_data}
    assert external_data["location"] == weights_path.name
    assert weights_path.read_bytes() == weight_bytes


def test_embed_passes_only_inputs_accepted_by_exported_graph() -> None:
    np = pytest.importorskip("numpy")

    class Session:
        received: dict | None = None

        def run(self, _outputs: None, inputs: dict) -> list:
            self.received = inputs
            shape = (*inputs["input_ids"].shape, 2)
            return [np.ones(shape, dtype=np.float32)]

    provider = object.__new__(ONNXProvider)
    provider._doc_prefix = ""
    provider._input_names = {"input_ids", "attention_mask"}
    provider._session = Session()
    provider._tokenizer = lambda *_args, **_kwargs: {
        "input_ids": np.ones((1, 512), dtype=np.int64),
        "attention_mask": np.ones((1, 512), dtype=np.int64),
        "token_type_ids": np.zeros((1, 512), dtype=np.int64),
    }

    embeddings = provider.embed(["test"])

    assert len(embeddings) == 1
    assert set(provider._session.received or {}) == {"input_ids", "attention_mask"}
