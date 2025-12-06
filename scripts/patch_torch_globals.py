
import torch
import numpy
try:
    # Attempt to whitelist the scalar type if torch supports safe_globals (PyTorch 2.4+)
    # The error path is 'numpy._core.multiarray.scalar'
    # We need to find the actual class reference for this.
    # In numpy 1.26, it might be numpy.core.multiarray.scalar or similar.
    # However, the error message specifically says 'numpy._core.multiarray.scalar'.
    # This implies the pickle stream refers to it that way.
    # Let's try to find where it lives.
    
    # Actually, simpler approach: just add the class if we can find it.
    # But often this specific path is an alias or internal.
    
    # Workaround: Monkey patch check_module_safe_globals or add to the list if possible.
    # But usually, just importing this module before torch.load is called might not be enough
    # if we don't modify the safe list.
    
    # Let's try to add it to torch.serialization.add_safe_globals if it exists.
    if hasattr(torch.serialization, 'add_safe_globals'):
        # We need the actual class. 
        # 'numpy._core.multiarray.scalar' usually maps to numpy.dtype or specific scalar types.
        # But wait, 'scalar' is a type?
        # Let's try adding common numpy scalar types.
        torch.serialization.add_safe_globals([
            numpy.core.multiarray.scalar,
            numpy.dtype,
            numpy.ndarray
        ])
        print("Patched torch.serialization.add_safe_globals with numpy types")
except Exception as e:
    print(f"Could not patch torch globals: {e}")

# Alternative: Disable weights_only=True if it is being used by default (PyTorch 2.6 does this!)
# We might need to monkeypatch torch.load to set weights_only=False by default if the user code doesn't specify it.
original_load = torch.load
def patched_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = patched_load
print("Monkey-patched torch.load to set weights_only=False by default")
