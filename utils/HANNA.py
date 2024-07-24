import torch.nn as nn
import torch
import torch.nn.functional as F

class HANNA(nn.Module):
    def __init__(self, Embedding_ChemBERT=384, nodes=96):
        super(HANNA, self).__init__()

        self.Embedding_ChemBERT = Embedding_ChemBERT # Pre-trained embeddings (E_i) from ChemBERTa-2
        self.nodes = nodes # Number of Nodes in HANNA

        # Component Embedding Network f_theta Input: E_i Output: f_theta(E_i)
        self.theta = nn.Sequential(
            nn.Linear(Embedding_ChemBERT, nodes),
            nn.SiLU(),
        )

        # Mixture Embedding Network f_alpha Input: C_i, Output: f_alpha(C_i)
        # nodes+2 is needed for concatenating T and x_i to the embedding f_theta(E_i)
        self.alpha = nn.Sequential(
            nn.Linear(nodes+2, nodes),
            nn.SiLU(),
            nn.Linear(nodes, nodes),
            nn.SiLU(),
        )

        # Property Network Input f_phi Input: C_mix, Output: g^E_NN 
        self.phi = nn.Sequential(
            nn.Linear(nodes, nodes),
            nn.SiLU(),
            nn.Linear(nodes, 1)
        )

    def forward(self, temperature, mole_fractions, E_i):
        # Determine batch_size (B) and number of components (N)
        batch_size, num_components, _ = E_i.shape # [B,N,E] E=384, ChemBERTa-2 embedding

        # Enable gradient tracking to use autograd
        E_i.requires_grad_(True)
        temperature.requires_grad_(True) # Standardized temperature
        mole_fractions.requires_grad_(True) # x_1

        # Calculate remaining mole fraction for the Nth component (here: N=2)
        mole_fraction_N = 1 - mole_fractions.sum(dim=1, keepdim=True) # x_2=1-x_1 [B,1]
        mole_fractions_complete = torch.cat([mole_fractions, mole_fraction_N], dim=1) # [x_1,1-x_1], [B,2]

        # Reshape mole fraction and temperature
        mole_fractions_complete_reshaped = mole_fractions_complete.unsqueeze(-1) # [B,N,1]
        T_reshaped = temperature.view(batch_size, 1, 1).expand(-1, num_components, 1) # [B,N,1]

        # Fine-tuning of the component embeddings
        theta_E_i = self.theta(E_i) # [B,N,nodes]

        # Calculate cosine similarity between the two components
        cosine_sim = F.cosine_similarity(theta_E_i[:, 0, :], theta_E_i[:, 1, :], dim=1) #[B]
        # Calculate cosine distance between the two components
        cosine_distance = 1 - cosine_sim # [B]

        # Concatenate embedding with T and x_i
        c_i = torch.cat([T_reshaped, mole_fractions_complete_reshaped, theta_E_i], dim=-1) #[B,N,nodes+2]
        alpha_c_i = self.alpha(c_i) # [B,N,nodes]
        c_mix = alpha_c_i.sum(dim=1) # [B,nodes]
        gE_NN = self.phi(c_mix).squeeze(-1) # [B]

        # Apply cosine similarity adjustment
        correction_factor_mole_fraction = torch.prod(mole_fractions_complete, dim=1) # [B] x1*(1-x1) term
        gE = gE_NN * correction_factor_mole_fraction * cosine_distance  # [B] Adjust gE_NN with the physical constraints and calculate gE/RT

        # Compute (dgE/dx1)/RT
        dgE_dx1 = torch.autograd.grad(gE.sum(), mole_fractions, create_graph=True)[0] # [B,1]

        # ln gamma_i equation (binary mixture). Unsqueeze to adjust dimension to [B,1] for gE/RT
        ln_gamma_1 = gE.unsqueeze(1) + (1 - mole_fractions) * dgE_dx1 # [B,1]
        ln_gamma_2 = gE.unsqueeze(1) - mole_fractions * dgE_dx1 # [B,1]
        # Concatenate the ln_gammas
        ln_gammas = torch.cat([ln_gamma_1, ln_gamma_2], dim=1) # [B,2]

        return ln_gammas, gE
