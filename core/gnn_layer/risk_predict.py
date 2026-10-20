import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv

from neo4j import GraphDatabase
import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score
from core.kg_sna_layer.graph_builder import calculate_strict_fold_topology

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score, precision_recall_curve, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ==========================================
# 1. THE MODEL ARCHITECTURE
# ==========================================
class RiskEscalationGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, edge_dim=3):
        super(RiskEscalationGNN, self).__init__()
        self.edge_norm = torch.nn.BatchNorm1d(edge_dim)
        # GATv2Conv (Graph Attention) learns which neighboring nodes are the most dangerous.
        self.conv1 = GATv2Conv(num_node_features, hidden_channels, edge_dim=edge_dim)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, edge_dim=edge_dim)
        
        # Linear layer to classify the edge (Escalation vs. No Escalation)
        self.classifier = torch.nn.Linear((hidden_channels * 2) + edge_dim, 1)
        #self.classifier = torch.nn.Sequential(
        #    torch.nn.Linear((hidden_channels * 2) + edge_dim, 32),
        #    torch.nn.ReLU(),
        #    torch.nn.Dropout(p=0.3),
        #    torch.nn.Linear(32, 1)
        #)

    def forwardV1(self, x, edge_index, edge_attr):
        # 1. Message Passing.
        edge_attr = self.edge_norm(edge_attr)
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        #x = F.dropout(x, p=0.5, training=self.training)        
        x = self.conv2(x, edge_index, edge_attr)
        
        # 2. Edge Classification (Combine features of Node A and Node B)
        row, col = edge_index
        edge_features = torch.cat([x[row], x[col], edge_attr], dim=1)
        
        # 3. Predict the probability of escalation
        return self.classifier(edge_features)

    def forward(self, x, edge_index, edge_attr, cls_edge_index=None, cls_edge_attr=None):
        mp_attr = self.edge_norm(edge_attr)
        x = F.relu(self.conv1(x, edge_index, mp_attr))
        x = self.conv2(x, edge_index, mp_attr)
        if cls_edge_index is None:
            cls_edge_index, cls_attr = edge_index, mp_attr
        else:
            cls_attr = self.edge_norm(cls_edge_attr)      # use only in eval mode
        row, col = cls_edge_index
        return self.classifier(torch.cat([x[row], x[col], cls_attr], dim=1))
    
class FocalLossV1(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Calculate standard BCE
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate the probabilities
        pt = torch.exp(-BCE_loss) 
        
        # Apply the focal loss formula
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return 

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super().__init__(); self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * (1 - pt) ** self.gamma * bce
        return loss.mean() if self.reduction == 'mean' else loss.sum() if self.reduction == 'sum' else loss
# ==========================================
# 2. DATA EXTRACTION
# ==========================================
def fetch_graph_from_neo4j(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with driver.session() as session:
        # 1. Fetch Nodes and SNA metrics
     
        node_query = """
        MATCH (i) 
        WHERE i.name IS NOT NULL
        RETURN i.name AS name
        ORDER BY i.name ASC
        """
        nodes = session.run(node_query).data()
        
        # Safety check for empty database
        if not nodes:
            raise ValueError("No nodes found in Neo4j! Make sure your graph is populated.")
            
        node_mapping = {node['name']: idx for idx, node in enumerate(nodes)}

        # Create a reverse mapping so we remember who is who!
        reverse_mapping = {idx: node['name'] for idx, node in enumerate(nodes)}
        
        # Handle potential None values if SNA metrics haven't been calculated properly
        #x_enriched_d = torch.tensor([[n['rep'] or 0.0, n['betw'] or 0.0, n['eigen'] or 0.0] for n in nodes], dtype=torch.float)
        x_enriched = torch.ones((len(nodes), 1), dtype=torch.float)
        x_baseline = torch.ones((len(nodes), 1), dtype=torch.float)

        # 2. Fetch Edges, Taxonomy Weights, AND Topological Edge Features
      
        edge_queryv1 = """
        MATCH (a)-[r]->(b) 
        WHERE type(r) <> 'SNA_TIE_STRENGTH'
        OPTIONAL MATCH (a)-[meta:SNA_TIE_STRENGTH]->(b)
        RETURN a.name AS source, 
               b.name AS target, 
               r.risk_weight AS weight,
               type(r) AS rel_type,  
               meta.common_neighbors AS cn,
               meta.jaccard AS jac,
               meta.adamic_adar AS aa
        ORDER BY a.name ASC, b.name ASC, type(r) ASC
        """
        edge_query = """
            MATCH (a)-[r]->(b)
            WHERE type(r) <> 'SNA_TIE_STRENGTH'
            RETURN a.name AS source, b.name AS target, r.risk_weight AS weight,
                type(r) AS rel_type, r.log_id AS log_id, r.taxonomy AS taxonomy
            ORDER BY a.name ASC, b.name ASC, type(r) ASC, r.log_id ASC
            """

        ESCALATION_CLASSES = {"THREAT", "EXTREMIST", "RADICALISM"}
        log_ids, y_w38 = [], []
        edges = session.run(edge_query).data()

        taxonomy_mapping = {
            'AFFILIATED_WITH': 0, 'ASSAULTS': 1, 'ASSOCIATED_WITH': 2, 
            'COMMUNICATES_WITH': 3, 'CONTAINS': 4, 'DISTRIBUTES': 5, 
            'EXCHANGES': 6, 'LOCATED_IN': 7, 'POSSESSES': 8, 
            'SELF_HARM': 9, 'THREATENS': 10
        }

        num_classes = len(taxonomy_mapping)
        
        edge_index = []
        labels = [] 
        edge_features_enriched = []
        edge_features_baseline = []

        edge_weights_list = []
        edge_taxonomy_list = []

        for e in edges:
            src = node_mapping.get(e['source'])
            tgt = node_mapping.get(e['target'])

            # 🛑 THE FIX: Ignore the edge entirely if the source and target are the exact same node!
            if src == tgt:
                continue
            
            if src is not None and tgt is not None:
                edge_index.append([src, tgt])
                weight = float(e['weight']) if e['weight'] is not None else 1.0

                # Target Ground Truth
                labels.append([1.0 if weight >= 3.8 else 0.0])

                # Capture the raw severity weight to power our topological math!
                #edge_weights_list.append(weight) # <--- CAPTURE WEIGHT
                labels.append([1.0 if e['taxonomy'] in ESCALATION_CLASSES else 0.0])   # primary label
                y_w38.append([1.0 if weight >= 3.8 else 0.0])                           # sensitivity label
                log_ids.append(e['log_id'])
                edge_features_enriched.append([0.0, 0.0, 0.0])    # features are computed per fold
                edge_features_baseline.append([0.0, 0.0, 0.0])

                #Capture taxonomy
                one_hot = [0.0] * num_classes
                tax_label = e['rel_type']
                if tax_label in taxonomy_mapping:
                    one_hot[taxonomy_mapping[tax_label]] = 1.0
                edge_taxonomy_list.append(one_hot)
                    
                # Extract the 3 Edge Features (default to 0.0 if Neo4j returns None)
                cn = float(e['cn'] or 0.0)
                jac = float(e['jac'] or 0.0)
                aa = float(e['aa'] or 0.0)
                
                # Append to the enriched tracker
                edge_features_enriched.append([cn, jac, aa])
                
                # The baseline gets zeros so the GNN doesn't crash, but gains no predictive advantage
                edge_features_baseline.append([0.0, 0.0, 0.0])

        # 3. Convert all lists to PyTorch Tensors
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous() 
        y = torch.tensor(labels, dtype=torch.float)
        edge_weight_tensor = torch.tensor(edge_weights_list, dtype=torch.float) # <--- NEW TENSOR
        edge_taxonomy_tensor = torch.tensor(edge_taxonomy_list, dtype=torch.float) # <--- NEW TENSOR
        
        edge_attr_enriched = torch.tensor(edge_features_enriched, dtype=torch.float)
        edge_attr_baseline = torch.tensor(edge_features_baseline, dtype=torch.float)

        uniq = {l: i for i, l in enumerate(sorted(set(log_ids), key=str))}
        log_id_t = torch.tensor([uniq[l] for l in log_ids], dtype=torch.long)
        y_w38_t = torch.tensor(y_w38, dtype=torch.float)

        # 4. Construct the Final Data Objects
        data_enriched = Data(x=x_enriched, edge_index=edge_index, y=y, edge_attr=edge_attr_enriched, edge_weight=edge_weight_tensor, taxonomy=edge_taxonomy_tensor)
        data_baseline = Data(x=x_baseline, edge_index=edge_index, y=y, edge_attr=edge_attr_baseline, edge_weight=edge_weight_tensor, taxonomy=edge_taxonomy_tensor)

        for d in (data_enriched, data_baseline):
            d.log_id, d.y_w38 = log_id_t, y_w38_t

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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    #criterion = torch.nn.BCEWithLogitsLoss()
    #criterion = FocalLoss(alpha=0.25, gamma=2.0)
    criterion = FocalLoss(alpha=0.5, gamma=2.0)

    history = {'loss': [], 'acc': []}
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad() 
        out = model(data.x, data.edge_index, data.edge_attr)
        
        # --- THE FINAL FIX ---
        # Squeeze BOTH so they are exactly 1D arrays of shape [Edges]
        out_flat = out.squeeze()
        y_true = data.y.float().squeeze() # <--- Add .squeeze() here!
        
        # Now both are exactly [3258] and perfectly aligned
        loss = criterion(out_flat, y_true)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            preds = (torch.sigmoid(out_flat) > 0.5).float()
            correct = (preds == y_true).float().sum()
            acc = correct / data.y.size(0)

        history['loss'].append(loss.item())
        history['acc'].append(acc.item())
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f}")
            
    return model, history

def evaluate_model(model, data, threshold=0.55, is_enriched=False):
    model.eval() 
    num_nodes = data.x.size(0)
    
    with torch.no_grad():
        # Reconstruct the structural metrics
        weights = data.edge_weight if is_enriched else None
        full_edge_attr = calculate_strict_fold_topology(
            data.edge_index, data.edge_index, num_nodes, edge_weights=weights
        )
        
        # Reconstruct the semantic taxonomy dimensions
        if is_enriched:
            full_edge_attr = torch.cat([full_edge_attr, data.taxonomy], dim=1)
            
        logits = model(data.x, data.edge_index, full_edge_attr)
        predictions = torch.sigmoid(logits)
        
        preds_binary = (predictions >= threshold).float().numpy()
        y_true = data.y.numpy()
        preds_probs = predictions.numpy()
        
        try:
            auc = roc_auc_score(y_true, preds_probs)
        except ValueError:
            auc = 0.5 
            
        recall = recall_score(y_true, preds_binary, zero_division=0)
        
        return auc, recall, y_true, preds_probs

def evaluate_with_kfold(model_template, data, n_splits=5, epochs=250, is_enriched=False):
    skf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=5, random_state=42)
    
    # Trackers
    all_histories = []
    oof_y_true = []
    oof_y_pred_probs = []

    y_array = data.y.cpu().numpy().flatten() 
    edge_indices = np.arange(len(y_array))
    num_nodes = data.x.size(0)

    for _, (train_idx, test_idx) in enumerate(skf.split(edge_indices, y_array)):
        
        # 1. Slice ONLY the raw topology (Nodes and Edges)
        train_edge_index = data.edge_index[:, train_idx]
        test_edge_index = data.edge_index[:, test_idx]

        # SLICE THE WEIGHTS FOR THE TRAINING SET
        train_edge_weights = data.edge_weight[train_idx] if is_enriched else None

        # 2. THE LEAK FIX: Calculate edge features dynamically on the fly!
        train_edge_attr = calculate_strict_fold_topology(
            train_edge_index, train_edge_index, num_nodes, edge_weights=train_edge_weights
        )
        test_edge_attr = calculate_strict_fold_topology(
            train_edge_index, test_edge_index, num_nodes, edge_weights=train_edge_weights
        ) 

        if is_enriched:
            # Slice the 11-dimensional taxonomy vectors for this specific fold split
            train_taxonomy = data.taxonomy[train_idx]
            test_taxonomy = data.taxonomy[test_idx]
            
            # Combine the 8 structural features with the 11 behavior classes via tensor concatenation
            train_edge_attr = torch.cat([train_edge_attr, train_taxonomy], dim=1)
            test_edge_attr = torch.cat([test_edge_attr, test_taxonomy], dim=1)

        # 3. Assemble Safe Data Objects
        train_data = Data(
            x=data.x,
            edge_index=train_edge_index,
            y=data.y[train_idx],
            edge_attr=train_edge_attr # Use the dynamically calculated attributes
        )

        test_data = Data(
            x=data.x,
            edge_index=test_edge_index,
            y=data.y[test_idx],
            edge_attr=test_edge_attr # Use the dynamically calculated attributes
        )

        # 4. Balance the Training Data strictly (Leaves Test Data untouched)
        balanced_train_data = graph_edge_oversample(train_data)

        # 5. Spawn a fresh, blank model
        fold_model = copy.deepcopy(model_template).to(data.x.device)
        fold_model.apply(reset_weights)

        # 6. Train on the Balanced Train Slice
        trained_fold_model, fold_history = train_model(fold_model, balanced_train_data, epochs=epochs)
        
        # ---> HERE IS WHERE YOU APPEND THE HISTORY <---
        all_histories.append(fold_history) 

        # 7. Predict on the unseen, imbalanced Test Slice
        trained_fold_model.eval()
        with torch.no_grad():
            logits = trained_fold_model(test_data.x, test_data.edge_index, test_data.edge_attr)
            probs = torch.sigmoid(logits).cpu().numpy()

        # 8. Save the honest predictions
        oof_y_true.append(test_data.y.cpu().numpy().flatten())
        oof_y_pred_probs.append(probs.flatten())

    # 9. Return concatenated arrays (Required for Scikit-Learn curve plotters!)
    return np.concatenate(oof_y_true), np.concatenate(oof_y_pred_probs), all_histories


# ==========================================
# 4. ACTIONABLE INTELLIGENCE (INFERENCE)
# ==========================================
def generate_threat_report(model, data, reverse_mapping, threshold=0.55, is_enriched=False):
    """
    Translates raw PyTorch tensor predictions back into human-readable alerts,
    dynamically reconstructing the edge feature space to avoid dimension mismatches.
    """
    model.eval()
    num_nodes = data.x.size(0)
    
    with torch.no_grad():
        # 1. Reconstruct the full structural topology features (6 or 8 dimensions)
        weights = data.edge_weight if is_enriched else None
        full_edge_attr = calculate_strict_fold_topology(
            data.edge_index, data.edge_index, num_nodes, edge_weights=weights
        )
        
        # 2. Append the 11-dimensional taxonomy vector if using the enriched model
        if is_enriched:
            full_edge_attr = torch.cat([full_edge_attr, data.taxonomy], dim=1)
            
        # 3. Pass the fully reconstructed attribute tensor to the model
        logits = model(data.x, data.edge_index, full_edge_attr)
        predictions = torch.sigmoid(logits).numpy() 
        
    alerts = []
    for i, prob_array in enumerate(predictions):
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

FEATURE_CONFIGS = {
    "A_struct":      dict(weighted=False, reltype=False),   # baseline, 6 features
    "B_struct+rel":  dict(weighted=True,  reltype=False),   # 8
    "C_struct+type": dict(weighted=False, reltype=True),    # 17
    "D_full":        dict(weighted=True,  reltype=True),    # 19
}

def best_f1_threshold(y_true, y_prob):
    if int(np.sum(y_true)) == 0:
        return 0.5
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    den = prec[:-1] + rec[:-1]
    f1 = np.divide(2 * prec[:-1] * rec[:-1], den, out=np.zeros_like(den), where=den > 0)
    return float(thr[int(np.argmax(f1))])

def _prf(y_t, p_t, tau):
    return precision_recall_fscore_support(y_t, (p_t >= tau).astype(int), average="binary", zero_division=0)[:3]

def _features(data, ref_idx, tgt_idx, weighted):
    w = data.edge_weight[ref_idx] if weighted else None
    return calculate_strict_fold_topology(data.edge_index[:, ref_idx], data.edge_index[:, tgt_idx],
                                          data.x.size(0), edge_weights=w)

def _oof_train_features(data, fit, g, y, weighted, inner_k, seed):
    """Features of each training edge from a reference graph that excludes its own narrative fold."""
    out = torch.zeros(len(fit), 8 if weighted else 6)
    inner = StratifiedGroupKFold(n_splits=inner_k, shuffle=True, random_state=seed)
    for tr, te in inner.split(np.arange(len(fit)), y[fit], g[fit]):
        out[te] = _features(data, fit[tr], fit[te], weighted)
    return out

def _attr(feats, data, cfg, part, idx):
    parts = [feats[cfg["weighted"]][part]] + ([data.taxonomy[idx]] if cfg["reltype"] else [])
    return torch.cat(parts, dim=1)

def _record(rows, preds, name, fold, rep, y, val, p_val, test, p_test):
    tau = best_f1_threshold(y[val], p_val)
    yt, has_pos = y[test], y[test].sum() > 0
    pr, rc, f1 = _prf(yt, p_test, tau)
    rows.append(dict(config=name, fold=fold, rep=rep, tau=tau, n_test=len(test), pos_test=int(yt.sum()),
                     auc=roc_auc_score(yt, p_test) if has_pos else np.nan,
                     ap=average_precision_score(yt, p_test) if has_pos else np.nan,
                     prevalence=float(yt.mean()), precision=pr, recall=rc, f1=f1,
                     f1_oracle=_prf(yt, p_test, best_f1_threshold(yt, p_test))[2]))   # test-tuned, for reference only
    for split, idx, p in (("val", val, p_val), ("test", test, p_test)):
        preds.append(pd.DataFrame(dict(config=name, fold=fold, rep=rep, split=split, edge=idx, y=y[idx], p=p)))

def evaluate_grouped_cv(data, groups, configs=FEATURE_CONFIGS, n_splits=5, n_repeats=5, epochs=150,
                        hidden=64, val_splits=5, inner_k=5, seed=42, with_lr=True):
    y = data.y.cpu().numpy().ravel().astype(int)
    g = np.asarray(groups.cpu() if torch.is_tensor(groups) else groups)
    rows, preds, hist = [], [], {n: [] for n in configs}

    for rep in range(n_repeats):
        outer = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        for k, (trval, test) in enumerate(outer.split(np.arange(len(y)), y, g)):
            fold = rep * n_splits + k
            inner = StratifiedGroupKFold(n_splits=val_splits, shuffle=True, random_state=seed + 1000 + fold)
            fit_rel, val_rel = next(inner.split(trval, y[trval], g[trval]))
            fit, val = trval[fit_rel], trval[val_rel]
            feats = {w: dict(fit=_oof_train_features(data, fit, g, y, w, inner_k, seed + fold),
                             val=_features(data, fit, val, w), test=_features(data, fit, test, w))
                     for w in (False, True)}                      # computed once, shared by all configs

            for name, cfg in configs.items():
                tr = Data(x=data.x, edge_index=data.edge_index[:, fit], y=data.y[fit],
                          edge_attr=_attr(feats, data, cfg, "fit", fit))
                bal = graph_edge_oversample(tr)
                torch.manual_seed(seed + fold)
                model = RiskEscalationGNN(data.x.size(1), hidden, edge_dim=tr.edge_attr.size(1))
                model, h = train_model(model, bal, epochs=epochs); hist[name].append(h)
                model.eval()
                with torch.no_grad():
                    def score(idx, part):
                        out = model(bal.x, bal.edge_index, bal.edge_attr,
                                    cls_edge_index=data.edge_index[:, idx],
                                    cls_edge_attr=_attr(feats, data, cfg, part, idx))
                        return torch.sigmoid(out).cpu().numpy().ravel()
                    p_val, p_test = score(val, "val"), score(test, "test")
                _record(rows, preds, name, fold, rep, y, val, p_val, test, p_test)

                if with_lr:                                        # non-graph baseline, same features and splits
                    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
                    clf.fit(_attr(feats, data, cfg, "fit", fit).numpy(), y[fit])
                    _record(rows, preds, "LR_" + name, fold, rep, y,
                            val, clf.predict_proba(_attr(feats, data, cfg, "val", val).numpy())[:, 1],
                            test, clf.predict_proba(_attr(feats, data, cfg, "test", test).numpy())[:, 1])
            if with_lr:                                            # relation type alone
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
                clf.fit(data.taxonomy[fit].numpy(), y[fit])
                _record(rows, preds, "LR_type_only", fold, rep, y,
                        val, clf.predict_proba(data.taxonomy[val].numpy())[:, 1],
                        test, clf.predict_proba(data.taxonomy[test].numpy())[:, 1])
            print(f"fold {fold + 1}/{n_splits * n_repeats} done")
    return pd.DataFrame(rows), pd.concat(preds, ignore_index=True), hist