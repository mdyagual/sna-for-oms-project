import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from neo4j import GraphDatabase

from sklearn.metrics import roc_auc_score, recall_score


# ==========================================
# 1. THE MODEL ARCHITECTURE
# ==========================================
class RiskEscalationGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels):
        super(RiskEscalationGNN, self).__init__()
        # GATConv (Graph Attention) learns which neighboring nodes are the most dangerous.
        self.conv1 = GATConv(num_node_features, hidden_channels, edge_dim=1)
        self.conv2 = GATConv(hidden_channels, hidden_channels, edge_dim=1)
        
        # Linear layer to classify the edge (Escalation vs. No Escalation)
        self.classifier = torch.nn.Linear(hidden_channels * 2, 1)

    def forward(self, x, edge_index, edge_attr):
        # 1. Message Passing
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        
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
        node_query = "MATCH (i:Inmate) RETURN i.name AS name, i.confidence_score AS deg, i.betweenness AS betw, i.eigenvector AS eigen"
        nodes = session.run(node_query).data()
        
        # Safety check for empty database
        if not nodes:
            raise ValueError("No nodes found in Neo4j! Make sure your graph is populated.")
            
        node_mapping = {node['name']: idx for idx, node in enumerate(nodes)}

        #Create a reverse mapping so we remember who is who!
        reverse_mapping = {idx: node['name'] for idx, node in enumerate(nodes)}
        
        # Handle potential None values if SNA metrics haven't been calculated properly
        x_enriched = torch.tensor([[n['deg'] or 0.0, n['betw'] or 0.0, n['eigen'] or 0.0] for n in nodes], dtype=torch.float)
        x_baseline = torch.ones((len(nodes), 1), dtype=torch.float)

        # Fetch Edges and Taxonomy Weights
        edge_query = """MATCH (a:Inmate)-[r]->(b:Inmate) 
        WHERE type(r) <> 'SNA_TIE_STRENGTH'
        RETURN a.name AS source, b.name AS target, r.risk_weight AS weight"""
        edges = session.run(edge_query).data()
        
        edge_index = []
        edge_weight_enriched = []
        labels = [] 

        for e in edges:
            src = node_mapping.get(e['source'])
            tgt = node_mapping.get(e['target'])
            
            if src is not None and tgt is not None:
                edge_index.append([src, tgt])
                weight = float(e['weight']) if e['weight'] is not None else 1.0
                edge_weight_enriched.append([weight])

                # CHANGE THIS LINE: Lower the threshold from 4.0 to 2.5
                labels.append([1.0 if weight >= 4.0 else 0.0])

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() 
        edge_weight_enriched = torch.tensor(edge_weight_enriched, dtype=torch.float)
        edge_weight_baseline = torch.ones((len(edge_index[0]), 1), dtype=torch.float) 
        y = torch.tensor(labels, dtype=torch.float)

        data_enriched = Data(x=x_enriched, edge_index=edge_index, edge_attr=edge_weight_enriched, y=y)
        data_baseline = Data(x=x_baseline, edge_index=edge_index, edge_attr=edge_weight_baseline, y=y)
        
        return data_baseline, data_enriched, reverse_mapping

# ==========================================
# 3. TRAINING AND EVALUATION
# ==========================================
def train_model(model, data, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    
    # --- THE FIX: CALCULATE CLASS IMBALANCE WEIGHT ---
    # Count how many positive (escalation) and negative edges exist
    num_positives = data.y.sum().item()
    num_negatives = len(data.y) - num_positives
    
    # Calculate ratio (e.g., if 980 negatives and 20 positives, weight is 49.0)
    # Added a fallback to 1.0 just in case there are 0 positives in a test run
    weight_val = num_negatives / num_positives if num_positives > 0 else 1.0
    pos_weight = torch.tensor([weight_val])
    
    # Switch to BCEWithLogitsLoss and apply the weight
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad() 
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f}")
            
    return model

def evaluate_model(model, data, threshold=0.15):
    model.eval() 
    with torch.no_grad():
        # 1. Get raw logits from the model
        logits = model(data.x, data.edge_index, data.edge_attr)
        
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
# ==========================================
# 4. ACTIONABLE INTELLIGENCE (INFERENCE)
# ==========================================
def generate_threat_report(model, data, reverse_mapping, threshold=0.70):
    """
    Translates raw PyTorch tensor predictions back into human-readable alerts.
    """
    model.eval()
    with torch.no_grad():
        # Get raw logits and apply sigmoid
        logits = model(data.x, data.edge_index, data.edge_attr)
        predictions = torch.sigmoid(logits).numpy() # <--- Added Sigmoid
        
    alerts = []
    # Loop through every edge in the network
    for i in range(len(predictions)):
        prob = predictions[i][0]
        
        # If the GNN thinks this edge is dangerous...
        if prob >= threshold:
            src_id = data.edge_index[0][i].item()
            tgt_id = data.edge_index[1][i].item()
            
            # Translate IDs back to names
            src_name = reverse_mapping[src_id]
            tgt_name = reverse_mapping[tgt_id]
            
            alerts.append({
                "source": src_name,
                "target": tgt_name,
                "probability": prob * 100
            })
            
    # Sort alerts by highest risk first
    alerts = sorted(alerts, key=lambda x: x['probability'], reverse=True)
    return alerts