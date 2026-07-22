import sys
import os
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_update.model import LLaDAMoEKV, SMALL_CFG
from model_update.generate import generate_des, LogProbVerifier

def test_des():
    print("Initializing model with SMALL_CFG...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LLaDAMoEKV(SMALL_CFG).to(device)
    model.eval()

    prompt_ids = torch.randint(0, SMALL_CFG.VS, (1, 16)).to(device)
    
    verifier = LogProbVerifier(model)
    
    print("Running generate_des...")
    out = generate_des(
        model=model,
        prompt_ids=prompt_ids,
        verifier=verifier,
        gen_length=32,
        steps=32,
        block_length=16,
        N=12,
        M=3,
        k_candidates=(4, 5, 6, 7),
        max_des_steps=2,
        temperature=0.8
    )
    
    print(f"Output shape: {out.shape}")
    assert out.shape == (1, 32), f"Expected shape (1, 32), got {out.shape}"
    print("Sanity check passed!")

if __name__ == "__main__":
    test_des()
