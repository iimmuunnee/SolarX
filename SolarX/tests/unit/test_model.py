"""Unit tests for LSTM model."""
import pytest
import torch
import numpy as np


def test_model_forward_pass(sample_lstm_model, sample_sequences):
    """Test that model forward pass produces correct output shape."""
    model = sample_lstm_model
    X, y = sample_sequences

    X_tensor = torch.tensor(X, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        output = model(X_tensor)

    # Output should be (batch_size, 1)
    assert output.shape == (X.shape[0], 1), f"Output shape mismatch: {output.shape}"


def test_model_parameters_trainable(sample_lstm_model):
    """Test that model parameters are trainable."""
    model = sample_lstm_model

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert trainable_params > 0, "Model should have trainable parameters"


def test_model_reproducibility(sample_lstm_model, sample_sequences):
    """Test that model produces same output for same input."""
    model = sample_lstm_model
    X, _ = sample_sequences

    X_tensor = torch.tensor(X, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        output1 = model(X_tensor)
        output2 = model(X_tensor)

    torch.testing.assert_close(output1, output2, msg="Model should be deterministic in eval mode")


def test_model_batch_independence(sample_lstm_model, sample_sequences):
    """Test that different batch sizes produce consistent per-sample results."""
    model = sample_lstm_model
    X, _ = sample_sequences

    # Process first sample individually
    X_single = torch.tensor(X[0:1], dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        output_single = model(X_single)

        # Process full batch
        X_batch = torch.tensor(X, dtype=torch.float32)
        output_batch = model(X_batch)

    # First output should match
    torch.testing.assert_close(output_single, output_batch[0:1], atol=1e-5, rtol=1e-5,
                                msg="Single and batch predictions should match")


def test_model_output_range(sample_lstm_model, sample_sequences):
    """Test that model outputs are reasonable (not NaN or Inf)."""
    model = sample_lstm_model
    X, _ = sample_sequences

    X_tensor = torch.tensor(X, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        output = model(X_tensor)

    assert not torch.isnan(output).any(), "Model output contains NaN"
    assert not torch.isinf(output).any(), "Model output contains Inf"


def test_model_gradient_flow(sample_lstm_model, sample_sequences):
    """Test that gradients flow through the model during training."""
    model = sample_lstm_model
    X, y = sample_sequences

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    model.train()

    # Forward pass
    output = model(X_tensor)
    loss = torch.nn.functional.mse_loss(output, y_tensor)

    # Backward pass
    loss.backward()

    # Check that gradients exist
    has_gradients = False
    for param in model.parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_gradients = True
            break

    assert has_gradients, "No gradients were computed"
