"""
Test tokenizer convenience methods on InterpretableModel.
"""

from mlxterp import InterpretableModel
from mlx_lm import load
import mlx.core as mx


def test_tokenizer_methods():
    """Test all tokenizer methods"""
    print("Loading model...")
    base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    print("\n" + "="*60)
    print("Testing Tokenizer Methods")
    print("="*60)

    # Test encode
    print("\n1. Testing encode()...")
    text = "Hello world"
    tokens = model.encode(text)
    print(f"   Text: '{text}'")
    print(f"   Tokens: {tokens}")
    assert isinstance(tokens, list), "encode() should return a list"
    assert len(tokens) > 0, "encode() should return non-empty list"
    print("   ✅ encode() works")

    # Test decode
    print("\n2. Testing decode()...")
    decoded = model.decode(tokens)
    print(f"   Tokens: {tokens}")
    print(f"   Decoded: '{decoded}'")
    assert isinstance(decoded, str), "decode() should return a string"
    assert "Hello" in decoded, "Decoded text should contain 'Hello'"
    print("   ✅ decode() works")

    # Test decode with mx.array
    print("\n3. Testing decode() with mx.array...")
    tokens_array = mx.array(tokens)
    decoded_from_array = model.decode(tokens_array)
    print(f"   mx.array({tokens})")
    print(f"   Decoded: '{decoded_from_array}'")
    assert decoded == decoded_from_array, "Should work with both list and mx.array"
    print("   ✅ decode() works with mx.array")

    # Test encode_batch
    print("\n4. Testing encode_batch()...")
    texts = ["Hello", "World", "Test"]
    token_lists = model.encode_batch(texts)
    print(f"   Texts: {texts}")
    for i, toks in enumerate(token_lists):
        print(f"   [{i}] {toks}")
    assert len(token_lists) == 3, "Should return 3 token lists"
    assert all(isinstance(t, list) for t in token_lists), "Each should be a list"
    print("   ✅ encode_batch() works")

    # Test token_to_str
    print("\n5. Testing token_to_str()...")
    token_id = tokens[0]
    token_str = model.token_to_str(token_id)
    print(f"   Token ID: {token_id}")
    print(f"   Token string: '{token_str}'")
    assert isinstance(token_str, str), "token_to_str() should return a string"
    print("   ✅ token_to_str() works")

    # Test vocab_size
    print("\n6. Testing vocab_size property...")
    vocab_size = model.vocab_size
    print(f"   Vocabulary size: {vocab_size}")
    assert vocab_size > 0, "Vocab size should be positive"
    print("   ✅ vocab_size works")

    # Test tokenizer attribute
    print("\n7. Testing direct tokenizer access...")
    assert model.tokenizer is not None, "Tokenizer should be accessible"
    assert model.tokenizer is tokenizer, "Should be same as original tokenizer"
    print("   ✅ Direct tokenizer access works")

    print("\n" + "="*60)
    print("✅ ALL TOKENIZER METHODS WORK!")
    print("="*60)

    return True


def test_tokenizer_with_tracing():
    """Test using tokenizer methods with tracing"""
    print("\n\n" + "="*60)
    print("Testing Tokenizer Methods with Tracing")
    print("="*60)

    from mlx_lm import load
    base_model, tokenizer = load('mlx-community/Llama-3.2-1B-Instruct-4bit')
    model = InterpretableModel(base_model, tokenizer=tokenizer)

    # Use tokenizer to analyze specific token positions
    text = "The capital of France is"
    tokens = model.encode(text)

    print(f"\nText: '{text}'")
    print("Token breakdown:")
    for i, token_id in enumerate(tokens):
        token_str = model.token_to_str(token_id)
        print(f"  Position {i}: {token_id:6d} -> '{token_str}'")

    # Trace and get activation at "France" position
    with model.trace(text) as trace:
        layer_5_out = trace.activations['model.model.layers.5']

    # Find "France" token
    france_pos = None
    for i, token_id in enumerate(tokens):
        if "France" in model.token_to_str(token_id):
            france_pos = i
            print(f"\n'France' token found at position {i}")
            break

    if france_pos is not None:
        france_activation = layer_5_out[0, france_pos, :]
        print(f"Activation at 'France': {france_activation.shape}")
        print("✅ Successfully extracted token-specific activation!")

    print("="*60)

    return True


if __name__ == "__main__":
    test_tokenizer_methods()
    test_tokenizer_with_tracing()
