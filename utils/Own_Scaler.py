from sklearn.preprocessing import StandardScaler
import numpy as np
# Definition of CustomScaler that scales all columns except the ones in columns_to_exclude (mole fraction here)
class CustomScaler:
    def __init__(self, Embedding_BERT):
        self.scaler = StandardScaler()
        self.columns_to_exclude = [1]  # Exclude the mole fraction column from scaling
        self.Embedding_BERT = Embedding_BERT

    def fit_transform(self, X, y=None):
        # Reshape the input data for scaling
        X = np.copy(X)
        N, n_comp, _ = X.shape # Number of samples, number of components
        X_reshaped = X.reshape(N * n_comp, 2+self.Embedding_BERT)
        # Scale only the columns that aren't in columns_to_exclude
        columns_to_scale = [col for col in range(X_reshaped.shape[1]) if col not in self.columns_to_exclude]
        X_reshaped[:, columns_to_scale] = self.scaler.fit_transform(X_reshaped[:, columns_to_scale]) # Call fit_transform from sklearn
        # Reshape the data back to original shape
        X_scaled = X_reshaped.reshape(N, n_comp, 2+self.Embedding_BERT)
        return X_scaled

    def transform(self, X, y=None):
        # Reshape the input data for scaling
        X = np.copy(X)
        N, n_comp, _ = X.shape # Number of samples, number of components 
        X_reshaped = X.reshape(N * n_comp, 2+self.Embedding_BERT) 
        # Scale only the columns that aren't in columns_to_exclude
        columns_to_scale = [col for col in range(X_reshaped.shape[1]) if col not in self.columns_to_exclude]
        X_reshaped[:, columns_to_scale] = self.scaler.transform(X_reshaped[:, columns_to_scale]) # Call transform from sklearn
        # Reshape the data back to original shape
        X_scaled = X_reshaped.reshape(N, n_comp, 2+self.Embedding_BERT)
        return X_scaled


    def inverse_transform(self, X):
        # Reshape the input data
        X = np.copy(X)
        N, n_comp, _ = X.shape
        X_reshaped = X.reshape(N * n_comp, 2 + self.Embedding_BERT)
        
        # Apply inverse transform only to the columns that were scaled
        columns_to_scale = [col for col in range(X_reshaped.shape[1]) if col not in self.columns_to_exclude]
        X_reshaped[:, columns_to_scale] = self.scaler.inverse_transform(X_reshaped[:, columns_to_scale]) #Call inverse_transform from sklearn
        
        # Reshape back to original shape
        X_inversed = X_reshaped.reshape(N, n_comp, 2 + self.Embedding_BERT)
        return X_inversed