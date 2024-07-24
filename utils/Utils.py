import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from rdkit import Chem
      
def predict(embedding_matrix, scaler, model, device): # Function to predict ln_gamma values
    # Transform input into the correct shape
    X_test_exp_processed = preprocess_input(embedding_matrix) # Shape: [num_samples, 2, 2+ BERT_Embedding]
    # Scale the processed input, each component is scaled separately, mole fraction is not scaled
    X_test_exp_scaled = scaler.transform(X_test_exp_processed) # Shape: [num_samples, 2, 2+ BERT_Embedding]
    # Split and reshape the input
    T_Test_exp, x_Test_exp, FP_Test_exp = split_and_reshape_input(X_test_exp_scaled) # Shapes: [num_samples], [num_samples, 1], [num_samples, 2, BERT_Embedding]
    # Convert to tensors
    T_temp_tensor = torch.tensor(T_Test_exp, dtype=torch.float32).to(device) # Shape: [num_samples]
    x_pred_tensor = torch.tensor(x_Test_exp, dtype=torch.float32).to(device) # Shape: [num_samples, 1]
    systems_FPs_tensor = torch.tensor(FP_Test_exp, dtype=torch.float32).to(device) # Shape: [num_samples, 2, BERT_Embedding]
    # Make predictions
    ln_gammas_pred, _ = model(T_temp_tensor, x_pred_tensor, systems_FPs_tensor) # Shape: [num_samples, 2]
    return x_pred_tensor.detach().numpy(), ln_gammas_pred.detach().numpy()
    

def preprocess_input( embedding_matrix, Embedding_BERT=384, num_components=2):
    # Embedding_matrix is organized as follows: [T, x1, emb1, emb2]
    # Shape of Embedding_matrix should be [num_samples, 770=2*384 + 2] in default case
    # num_samples = 200 in the default case
    # This funtion transforms the input into the following form: [T, x1, emb1] and [T, 1-x1, emb2]

    num_samples = embedding_matrix.shape[0]
    # Assert that the input shape matches our assumptions
    expected_columns = 1 + (num_components - 1) + num_components * Embedding_BERT
    assert embedding_matrix.shape[1] == expected_columns, f"Input shape doesn't match the expected shape based on Embedding_BERT. Expected {expected_columns} columns, got {embedding_matrix.shape[1]}"
    
    # Extract temperature column T
    Temp = embedding_matrix[:, :1]  # shape: [num_samples, 1]
    
    # Placeholder for the reshaped data
    reshaped_data = np.zeros((num_samples, num_components, Embedding_BERT + 2))
    
    # Fill in the temperature for all components
    reshaped_data[:, :, 0] = Temp[:, 0, None]
    
    for i in range(num_components):  
        # Mole fraction
        if i != num_components - 1: # For all components except the last one, here x1
            reshaped_data[:, i, 1] = embedding_matrix[:, i+1]
        else: # For the last component, here x2=1-x1
            reshaped_data[:, i, 1] = 1 - np.sum(reshaped_data[:, :-1, 1], axis=1) # Not used in model, but calculated for completeness
        
        # Extract the ChemBERT embeddings for the i-th component
        start_idx = 1 + num_components - 1 + i * Embedding_BERT # 1 for T, num_components - 1 for x1, and i * Embedding_BERT for the i-th component
        end_idx = start_idx + Embedding_BERT # End index for the i-th component
        reshaped_data[:, i, 2:] = embedding_matrix[:, start_idx:end_idx]
        
    return reshaped_data

def split_and_reshape_input(input_array):
    # input_array has a shape [batch_size, 2, total_feature_count]
    # Where total_feature_count = 1 (temperature) + 1 (mole fraction) + Embedding_BERT
    # Extract standardized temperature
    standardized_temp = input_array[:, 0, 0]  # Shape: [batch_size]
    # Extract mole fractions for N-1 components, here first component
    mole_fractions_N_minus_1 = input_array[:, 0, 1:2]  # Shape: [batch_size, 1]
    # Extract and reshape Feature Points of all components
    feature_points = input_array[:, :, 2:]
    return standardized_temp, mole_fractions_N_minus_1, feature_points

def canonicalize_smiles(smiles):
    # Canonicalize the SMILES
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
    return smiles

def get_smiles_embedding(smiles, model_version='DeepChem/ChemBERTa-77M-MTR', max_length=128):
    model = AutoModel.from_pretrained(pretrained_model_name_or_path=model_version, output_attentions=True) # Load ChemBERT model
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_version) # Load ChemBERT tokenizer
    # Build tokenizer with same padding as in training
    tokens = tokenizer(
                smiles,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )

    # Get the embeddings from CLS token
    with torch.no_grad():
        emb = model(
            tokens["input_ids"],
            tokens["attention_mask"]
        )["last_hidden_state"][:, 0, :].numpy()
    
    return emb

def create_embedding_matrix(smiles1, smiles2, T, x1_values=None):
    # Canonicalize the SMILES
    smiles1 = canonicalize_smiles(smiles1)
    smiles2 = canonicalize_smiles(smiles2)
    
    # Get embeddings for both SMILES
    emb1 = get_smiles_embedding(smiles1).flatten()
    emb2 = get_smiles_embedding(smiles2).flatten()
    
    # If x1_values is not provided, create a default range from 0 to 1 in 100 steps
    if x1_values is None:
        x1_values = np.linspace(0, 1, 100)
    
    # Create the matrix with varying x1
    embedding_matrix = []
    for x1 in x1_values:
        row = np.concatenate(([T, x1],emb1, emb2 ))
        embedding_matrix.append(row)
    
    return np.array(embedding_matrix)



