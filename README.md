# DSS-Analyse Mausversuch

Statistische Auswertung und Machine Learning Klassifikation für DSS-induzierte Darmentzündung.

## Dateien

- `analyse.py` - Hauptskript mit allen Analysen
- `dss_analyse.ipynb` - Jupyter Notebook Version
- `zeitverlaeufe.png` - Gewicht und Aktivität über Zeit
- `boxplots.png` - Vergleich der Dosisgruppen
- `cluster_wahl.png` - Optimierung der Cluster-Anzahl
- `clustering.png` - K-Means Ergebnisse
- `confusion_matrix.png` - Classifier Performance
- `decision_boundaries.png` - Entscheidungsgrenzen der Modelle
- `roc_curves.png` - ROC-Kurven

## Verwendung

```bash
python analyse.py
```

Oder öffne das Jupyter Notebook:
```bash
jupyter notebook dss_analyse.ipynb
```

## Ergebnisse

### Statistische Tests
- Tag 5: Signifikante Unterschiede (BWC p=0.003, VWR p=0.001)
- Tag 8: Hochsignifikant (BWC p<0.001, VWR p<0.001)
- Tag 13: Nur VWR noch signifikant (p=0.010)

### Clustering
- 3 Kategorien identifiziert: Gesund, Moderat, Schwer
- Silhouette Score: 0.586

### Classification
- Bestes Modell: Logistic Regression
- Accuracy: 99.4%
- Sensitivity: 94-100%
- Specificity: 99-100%

## Zusammenfassung

Die Analyse zeigt klare dosisabhängige Effekte. Die Laufradaktivität reagiert früher als das Körpergewicht. Ein Classifier kann die Belastungskategorien mit hoher Genauigkeit vorhersagen.
