"""
Analyse DSS-induzierte Darmentzündung
Datenauswertung Mausversuch
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             roc_curve, auc, silhouette_score)

sns.set_style('whitegrid')

# Globale Versuchsparameter
EVAL_DAYS = [5, 8, 13]

def load_data(filepath):
    df = pd.read_csv(filepath, sep='\t')
    print(f"Datensatz: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")
    print(f"Anzahl Tiere: {df['id'].nunique()}\n")
    return df

def run_exploratory_analysis(df):
    grouped = df.groupby(['DSS', 'day']).agg({
        'bwc': ['mean', 'sem'],
        'vwr': ['mean', 'sem']
    }).reset_index()

    grouped.columns = ['DSS', 'day', 'bwc_mean', 'bwc_sem', 'vwr_mean', 'vwr_sem']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for dss in sorted(df['DSS'].unique()):
        data = grouped[grouped['DSS'] == dss]
        ax1.plot(data['day'], data['bwc_mean'], 'o-', label=f'{dss}% DSS')
        ax1.fill_between(data['day'], 
                          data['bwc_mean'] - data['bwc_sem'],
                          data['bwc_mean'] + data['bwc_sem'], alpha=0.2)

    ax1.set_xlabel('Tag')
    ax1.set_ylabel('Body Weight Change (%)')
    ax1.set_title('Gewichtsverlauf')
    ax1.axhline(100, color='gray', linestyle='--', alpha=0.5)
    ax1.legend()
    ax1.grid(alpha=0.3)

    for dss in sorted(df['DSS'].unique()):
        data = grouped[grouped['DSS'] == dss]
        ax2.plot(data['day'], data['vwr_mean'], 'o-', label=f'{dss}% DSS')
        ax2.fill_between(data['day'], 
                          data['vwr_mean'] - data['vwr_sem'],
                          data['vwr_mean'] + data['vwr_sem'], alpha=0.2)

    ax2.set_xlabel('Tag')
    ax2.set_ylabel('Laufrad (rpm)')
    ax2.set_title('Aktivität')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('zeitverlaeufe.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Statistische Tests (Kruskal-Wallis):")
    for tag in EVAL_DAYS:
        data_tag = df[df['day'] == tag]
        
        gruppen_bwc = [data_tag[data_tag['DSS'] == d]['bwc'].values for d in sorted(df['DSS'].unique())]
        gruppen_vwr = [data_tag[data_tag['DSS'] == d]['vwr'].values for d in sorted(df['DSS'].unique())]
        
        h_bwc, p_bwc = stats.kruskal(*gruppen_bwc)
        h_vwr, p_vwr = stats.kruskal(*gruppen_vwr)
        
        print(f"Tag {tag}: BWC p={p_bwc:.4f}, VWR p={p_vwr:.4f}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for i, tag in enumerate(EVAL_DAYS):
        data = df[df['day'] == tag]
        sns.boxplot(data=data, x='DSS', y='bwc', ax=axes[0, i])
        axes[0, i].set_title(f'Tag {tag}')
        axes[0, i].set_ylabel('BWC (%)' if i == 0 else '')
        
        sns.boxplot(data=data, x='DSS', y='vwr', ax=axes[1, i])
        axes[1, i].set_ylabel('VWR (rpm)' if i == 0 else '')

    plt.tight_layout()
    plt.savefig('boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()

def extract_features_and_labels(df):
    X = df[['bwc', 'vwr']].values
    
    # Clustering als Labeling-Schritt
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = []
    silhouettes = []
    k_range = range(2, 8)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(k_range, inertias, 'o-')
    ax1.set_xlabel('Anzahl Cluster')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method')
    ax1.grid(alpha=0.3)
    
    ax2.plot(k_range, silhouettes, 'o-', color='orange')
    ax2.set_xlabel('Anzahl Cluster')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Analyse')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cluster_wahl.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nOptimale Cluster-Anzahl: k=3 (Silhouette: {max(silhouettes):.3f})")

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=20)
    cluster_labels = kmeans.fit_predict(X_scaled)
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    print("\nCluster-Zentren:")
    for i, c in enumerate(centers):
        print(f"  Cluster {i}: BWC={c[0]:.1f}%, VWR={c[1]:.1f} rpm")

    score = centers[:, 0] + centers[:, 1]
    order = np.argsort(score)
    mapping = {order[0]: 2, order[1]: 1, order[2]: 0}
    severity = np.array([mapping[c] for c in cluster_labels])
    df['severity'] = severity

    print(f"\nVerteilung: {df['severity'].value_counts().sort_index().to_dict()}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    scatter1 = ax1.scatter(df['bwc'], df['vwr'], c=cluster_labels, cmap='viridis', alpha=0.5, s=30)
    ax1.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, 
                edgecolors='black', linewidths=2, label='Zentren')
    ax1.set_xlabel('Body Weight Change (%)')
    ax1.set_ylabel('Laufrad (rpm)')
    ax1.set_title('K-Means Clustering')
    ax1.legend()
    ax1.grid(alpha=0.3)
    plt.colorbar(scatter1, ax=ax1, label='Cluster')

    colors = {0: 'green', 1: 'orange', 2: 'red'}
    labels_text = {0: 'Gesund', 1: 'Moderat', 2: 'Schwer'}
    for sev in [0, 1, 2]:
        mask = df['severity'] == sev
        ax2.scatter(df.loc[mask, 'bwc'], df.loc[mask, 'vwr'], 
                    c=colors[sev], label=labels_text[sev], alpha=0.5, s=30)

    ax2.set_xlabel('Body Weight Change (%)')
    ax2.set_ylabel('Laufrad (rpm)')
    ax2.set_title('Belastungskategorien')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('clustering.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df

def evaluate_metrics(y_test, y_pred, y_score, X, X_test, X_test_scaled, ergebnisse, best_model, scaler_ml):
    cm = confusion_matrix(y_test, y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Gesund', 'Moderat', 'Schwer'],
                yticklabels=['Gesund', 'Moderat', 'Schwer'])
    ax1.set_ylabel('Tatsächlich')
    ax1.set_xlabel('Vorhergesagt')
    ax1.set_title('Confusion Matrix')

    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='Greens', ax=ax2,
                xticklabels=['Gesund', 'Moderat', 'Schwer'],
                yticklabels=['Gesund', 'Moderat', 'Schwer'])
    ax2.set_ylabel('Tatsächlich')
    ax2.set_xlabel('Vorhergesagt')
    ax2.set_title('Confusion Matrix (normalisiert)')

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Classification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Gesund', 'Moderat', 'Schwer'],
                               digits=3))

    print("\nSensitivity und Specificity:")
    for i, label in enumerate(['Gesund', 'Moderat', 'Schwer']):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn
        
        sens = tp / (tp + fn) if (tp + fn) > 0 else 0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        print(f"{label}: Sens={sens:.3f}, Spec={spec:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    h = 0.5
    x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    y_min, y_max = X[:, 1].min() - 10, X[:, 1].max() + 10
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    for i, (name, res) in enumerate(ergebnisse.items()):
        Z = res['model'].predict(scaler_ml.transform(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)
        
        axes[i].contourf(xx, yy, Z, alpha=0.3, levels=2)
        axes[i].scatter(X_test[:, 0], X_test[:, 1], c=y_test, 
                       edgecolors='k', s=40, alpha=0.7)
        axes[i].set_xlabel('BWC (%)')
        axes[i].set_ylabel('VWR (rpm)')
        axes[i].set_title(f'{name}\nAcc: {res["accuracy"]:.3f}')
        axes[i].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('decision_boundaries.png', dpi=300, bbox_inches='tight')
    plt.close()

    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

    fpr = {}
    tpr = {}
    roc_auc = {}

    plt.figure(figsize=(8, 6))
    colors = ['green', 'orange', 'red']
    labels = ['Gesund', 'Moderat', 'Schwer']

    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                 label=f'{labels[i]} (AUC = {roc_auc[i]:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Kurven')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def train_and_evaluate_models(df):
    X = df[['bwc', 'vwr']].values
    y = df['severity'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler_ml = StandardScaler()
    X_train_scaled = scaler_ml.fit_transform(X_train)
    X_test_scaled = scaler_ml.transform(X_test)

    modelle = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM': SVC(kernel='linear', random_state=42, probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

    ergebnisse = {}

    print("\nModell-Performance:")
    for name, model in modelle.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
        
        ergebnisse[name] = {
            'model': model,
            'y_pred': y_pred,
            'accuracy': accuracy_score(y_test, y_pred),
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
        
        print(f"{name}: Test Acc={ergebnisse[name]['accuracy']:.3f}, CV={ergebnisse[name]['cv_mean']:.3f}")

    best_name = max(ergebnisse, key=lambda x: ergebnisse[x]['accuracy'])
    best_model = ergebnisse[best_name]['model']
    y_pred_best = ergebnisse[best_name]['y_pred']
    y_score_best = best_model.predict_proba(X_test_scaled)

    print(f"\nBestes Modell: {best_name}")
    print(f"Accuracy: {ergebnisse[best_name]['accuracy']:.3f}\n")

    evaluate_metrics(y_test, y_pred_best, y_score_best, X, X_test, X_test_scaled, ergebnisse, best_model, scaler_ml)

def main():
    df = load_data('testdata.txt')
    run_exploratory_analysis(df)
    df = extract_features_and_labels(df)
    train_and_evaluate_models(df)
    
    print("\n=== Analyse abgeschlossen ===")
    print("Grafiken gespeichert: zeitverlaeufe.png, boxplots.png, cluster_wahl.png,")
    print("                       clustering.png, confusion_matrix.png,")
    print("                       decision_boundaries.png, roc_curves.png")

if __name__ == '__main__':
    main()
