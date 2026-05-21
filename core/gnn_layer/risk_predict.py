import torch
import copy
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

from neo4j import GraphDatabase
import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score


# ==========================================
# 1. THE MODEL ARCHITECTURE
# ==========================================
class RiskEscalationGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(RiskEscalationGNN, self).__init__()
        # GATConv (Graph Attention) learns which neighboring nodes are the most dangerous.
        self.conv1 = GATConv(num_node_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        
        # Linear layer to classify the edge (Escalation vs. No Escalation)
        self.classifier = torch.nn.Linear(hidden_channels * 2,1)

    def forward(self, x, edge_index):
        # 1. Message Passing
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)        
        x = self.conv2(x, edge_index)
        
        # 2. Edge Classification (Combine features of Node A and Node B)
        row, col = edge_index

        edge_features = torch.cat([x[row], x[col]], dim=1)
        
        # 3. Predict the probability of escalation
        return self.classifier(edge_features)

# ==========================================
# 2. DATA EXTRACTION
# ==========================================
def fetch_graph_from_neo4j(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with driver.session() as session:
        # Fetch Nodes and SNA metrics
        node_query = "MATCH (i:Inmate) RETURN i.name AS name, i.reputation_score AS rep, i.betweenness AS betw, i.eigenvector AS eigen"        
        nodes = session.run(node_query).data()
        
        # Safety check for empty database
        if not nodes:
            raise ValueError("No nodes found in Neo4j! Make sure your graph is populated.")
            
        node_mapping = {node['name']: idx for idx, node in enumerate(nodes)}

        #Create a reverse mapping so we remember who is who!
        reverse_mapping = {idx: node['name'] for idx, node in enumerate(nodes)}
        
        # Handle potential None values if SNA metrics haven't been calculated properly
        x_enriched = torch.tensor([[n['rep'] or 0.0, n['betw'] or 0.0, n['eigen'] or 0.0] for n in nodes], dtype=torch.float)
        x_baseline = torch.ones((len(nodes), 1), dtype=torch.float)

        # Fetch Edges and Taxonomy Weights
        edge_query = """MATCH (a:Inmate)-[r]->(b:Inmate) 
        WHERE type(r) <> 'SNA_TIE_STRENGTH'
        RETURN a.name AS source, b.name AS target, r.risk_weight AS weight"""
        edges = session.run(edge_query).data()
        
        edge_index = []
        labels = [] 

        for e in edges:
            src = node_mapping.get(e['source'])
            tgt = node_mapping.get(e['target'])
            
            if src is not None and tgt is not None:
                edge_index.append([src, tgt])
                weight = float(e['weight']) if e['weight'] is not None else 1.0

                #This line is high linked to the severity level set in the Taxonomy Detection
                labels.append([1.0 if weight >= 4.0 else 0.0])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() 
        y = torch.tensor(labels, dtype=torch.float)

        data_enriched = Data(x=x_enriched, edge_index=edge_index, y=y)
        data_baseline = Data(x=x_baseline, edge_index=edge_index, y=y)
        
        return data_baseline, data_enriched, reverse_mapping

# ==========================================
# 3. TRAINING AND EVALUATION
# ==========================================
# Helper to completely wipe a model's memory
def reset_weights(m):
    if hasattr(m, 'reset_parameters'):
        m.reset_parameters()

def graph_edge_oversample(train_data):
    """
    Oversamples the minority class strictly within a PyTorch Geometric graph.
    Uses random sampling with replacement to achieve a perfect 50/50 mathematical balance.
    """
    # 1. Create 1D masks to identify classes safely
    minority_mask = (train_data.y == 1).squeeze()
    majority_mask = (train_data.y == 0).squeeze()

    # Isolate minority data
    minority_edges = train_data.edge_index[:, minority_mask]
    minority_labels = train_data.y[minority_mask]
    minority_attrs = train_data.edge_attr[minority_mask] if train_data.edge_attr is not None else None

    num_majority = majority_mask.sum().item()
    num_minority = minority_mask.sum().item()

    # Safety catch: Stop if no minority edges exist or if already balanced
    if num_minority == 0 or num_minority >= num_majority:
        return train_data

    # 2. Calculate exactly how many edges are needed to reach a 50/50 split
    num_to_add = num_majority - num_minority
    
    # Randomly select indices from the minority pool to duplicate (with replacement)
    indices_to_sample = torch.randint(0, num_minority, (num_to_add,))

    # 3. Generate the synthetic edge structure
    sampled_minority_edges = minority_edges[:, indices_to_sample]
    sampled_minority_labels = minority_labels[indices_to_sample]

    # 4. Concatenate the duplicated data to the original training data
    new_edge_index = torch.cat([train_data.edge_index, sampled_minority_edges], dim=1)
    new_y = torch.cat([train_data.y, sampled_minority_labels], dim=0)
    
    new_edge_attr = None
    if train_data.edge_attr is not None:
        sampled_minority_attrs = minority_attrs[indices_to_sample]
        new_edge_attr = torch.cat([train_data.edge_attr, sampled_minority_attrs], dim=0)

    # 5. Build and return the newly balanced Training Data object
    balanced_train_data = Data(
        x=train_data.x, 
        edge_index=new_edge_index, 
        edge_attr=new_edge_attr,
        y=new_y
    )
    
    return balanced_train_data

def train_model(model, data, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    loss_history = []
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad() 
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
            
    return model, loss_history

def evaluate_model(model, data, threshold=0.55):
    model.eval() 
    with torch.no_grad():
        # 1. Get raw logits from the model
        logits = model(data.x, data.edge_index)
        
        # 2. Manually apply Sigmoid to get probabilities (0.0 to 1.0)
        predictions = torch.sigmoid(logits)
        
        # 3. Use our new, highly-sensitive threshold
        preds_binary = (predictions >= threshold).float().numpy()
        y_true = data.y.numpy()
        preds_probs = predictions.numpy()
        
        try:
            auc = roc_auc_score(y_true, preds_probs)
        except ValueError:
            auc = 0.5 
            
        recall = recall_score(y_true, preds_binary, zero_division=0)
        
        return auc, recall, y_true, preds_probs
    
def evaluate_with_kfold(model_template, data, n_splits=5, epochs=250):
    skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=5, random_state=42)
    
    y_array = data.y.cpu().numpy().flatten() 
    edge_indices = np.arange(len(y_array))

    oof_y_true = []
    oof_y_pred_probs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(edge_indices, y_array)):
        
        # 1. Slice the Graph for the current fold
        train_data = Data(
            x=data.x,
            edge_index=data.edge_index[:, train_idx],
            y=data.y[train_idx]
        )

        test_data = Data(
            x=data.x,
            edge_index=data.edge_index[:, test_idx],
            y=data.y[test_idx]
        )

        # 2. Balance the Training Data strictly (Leaves Test Data untouched)
        balanced_train_data = graph_edge_oversample(train_data)

        # 3. Spawn a fresh, blank model
        fold_model = copy.deepcopy(model_template)
        fold_model.apply(reset_weights)

        # 4. Train on the Balanced Train Slice
        # (Passing balanced_train_data instead of train_data)
        trained_fold_model, _ = train_model(fold_model, balanced_train_data, epochs=epochs)

        # 5. Predict on the unseen, imbalanced Test Slice
        trained_fold_model.eval()
        with torch.no_grad():
            logits = trained_fold_model(test_data.x, test_data.edge_index)
            probs = torch.sigmoid(logits).cpu().numpy()

        # 6. Save the honest predictions
        oof_y_true.append(test_data.y.cpu().numpy().flatten())
        oof_y_pred_probs.append(probs.flatten())

    return oof_y_true, oof_y_pred_probs

# ==========================================
# 4. ACTIONABLE INTELLIGENCE (INFERENCE)
# ==========================================
def generate_threat_report(model, data, reverse_mapping, threshold=0.55):
    """
    Translates raw PyTorch tensor predictions back into human-readable alerts.
    """
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index)
        predictions = torch.sigmoid(logits).numpy() 
        
    alerts = []
    # Loop through every edge in the network
    for i, prob_array in enumerate(predictions):
        # FIX 1: Extract the scalar float from the numpy array!
        prob = prob_array[0]        
        
        if prob >= threshold:
            src_id = data.edge_index[0][i].item()
            tgt_id = data.edge_index[1][i].item()
            
            src_name = reverse_mapping[src_id]
            tgt_name = reverse_mapping[tgt_id]
            
            alerts.append({
                "source": src_name,
                "target": tgt_name,
                "probability": prob * 100
            })
            
    alerts = sorted(alerts, key=lambda x: x['probability'], reverse=True)
    return alerts