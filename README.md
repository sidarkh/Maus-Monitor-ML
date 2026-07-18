# Maus-Monitor-ML: Wissenschaftlicher Machine-Learning-Kurs

Dieses Repository enthält ein vollständiges, wissenschaftlich fundiertes Machine-Learning-Forschungsprojekt zur Klassifikation und Analyse von DSS-induzierter Darmentzündung in einem Mausversuch. 

Das Projekt dient als umfassender, phasenbasierter Leitfaden, der alle Schritte von den biologischen und statistischen Grundlagen bis hin zu komplexen Ensemble-Modellen, Erklärbarkeit (SHAP/LIME) und statistischer Modellvalidierung in genau **30 aufeinander aufbauenden Jupyter Notebooks** dokumentiert.

---

## Biologischer Hintergrund und Datensatz

Im Rahmen der Erforschung entzündlicher Darmerkrankungen (Colitis) erhalten Mäuse das Polysaccharid **Dextransulfat-Natrium (DSS)** über das Trinkwasser in drei verschiedenen Dosisgruppen:
- **0** = Kontrollgruppe (0% DSS)
- **1** = Niedrige Dosis (1.0% DSS)
- **2** = Hohe Dosis (1.5% DSS)

Über einen Verlauf von 14 Tagen (Tag 0 bis Tag 13) werden täglich zwei zentrale Variablen erhoben:
1. **`bwc` (Body Weight Change)**: Die prozentuale Gewichtsänderung bezogen auf den Ausgangswert an Tag 0 ($BWC_{i,0} = 100.0\%$).
2. **`vwr` (Voluntary Wheel Running)**: Die freiwillige Laufradaktivität in Radumdrehungen pro Minute (rpm) während der aktiven Nachtphase.

Durch das Erreichen klinisch-ethischer Abbruchkriterien ( humane Endpunkte) werden schwer belastete Tiere vorzeitig euthanasiert. Dies führt zu strukturell fehlenden Werten (*Missing Not At Random*, MNAR), was eine besondere Herausforderung für die Datenvorverarbeitung darstellt.

---

## Struktur des 30-Phasen-Lehrplans

Jedes Notebook ist wissenschaftlich aufgebaut und enthält: *Einführung, Lernziele, Theorie, mathematische Grundlagen, intuitive Erklärung, Python-Implementierung, Visualisierung, Interpretation, Zwischenfazit, Quizfragen, Zusammenfassung, Hausaufgabe und weiterführende Literatur*.

### Übersicht der Notebooks:
1. **[01_Project_Overview.ipynb](file:///Users/sidarkhalid/Downloads/Mäuse-Test-Daten/01_Project_Overview.ipynb)** – Biologische Grundlagen, Versuchsaufbau & Phasenstruktur.
2. **02_Data_Loading.ipynb** – Datenimport, Tabulator-Trennung & Pandas-Grundlagen.
3. **03_Data_Understanding.ipynb** – Strukturierte Dateninspektion & deskriptive Dimensionen.
4. **04_Data_Cleaning.ipynb** – Handhabung von fehlenden Werten (MNAR) & Duplikatsprüfung.
5. **05_Exploratory_Data_Analysis.ipynb** – Datenvisualisierung der Verläufe und Gruppenunterschiede.
6. **06_Univariate_Statistics.ipynb** – Lage- und Streuungsmaße, Schiefe & Wölbung (Kurtosis).
7. **07_Bivariate_Statistics.ipynb** – Korrelationsanalysen (Pearson, Spearman, Kendall).
8. **08_Hypothesis_Testing.ipynb** – Parametrische und nicht-parametrische Gruppenvergleiche.
9. **09_Time_Series_Analysis.ipynb** – Zeitreihenanalyse & Autokorrelationsstrukturen pro Tier.
10. **10_Feature_Engineering.ipynb** – Rollierende Fenster, Differenzen & Akkumulationen.
11. **11_Data_Preprocessing.ipynb** – Feature-Skalierung & Stratifizierter Train/Test-Split.
12. **12_Clustering.ipynb** – Unüberwachte Segmentierung (K-Means, Hierarchisch, DBSCAN, GMM).
13. **13_Cluster_Interpretation.ipynb** – Cluster-Validierung (Silhouette Score, Davies-Bouldin) & Profilierung.
14. **14_Classification_Baseline.ipynb** – Dummy-Modelle als systemische Untergrenze.
15. **15_Logistic_Regression.ipynb** – Mathematische Herleitung, Log-Loss-Funktion & Training.
16. **16_Linear_Discriminant_Analysis.ipynb** – Bayes'sche Entscheidungstheorie & Diskriminanzanalysen.
17. **17_KNearestNeighbors.ipynb** – Distanzbasierte Klassifikation & Hyperparameter $k$.
18. **18_Support_Vector_Machine.ipynb** – Margin-Maximierung, Kernel-Trick & SVM-Klassifikation.
19. **19_Decision_Tree.ipynb** – Entropie, Gini-Index & Pruning-Strategien.
20. **20_Random_Forest.ipynb** – Bagging-Verfahren, Out-of-Bag Error & Feature Importance.
21. **21_Gradient_Boosting.ipynb** – Boosting-Prinzipien & sequentielle Residuenminimierung.
22. **22_XGBoost.ipynb** – Extremes Gradient Boosting & Regularisierungs-Terme.
23. **23_Neural_Network.ipynb** – Künstliche Neuronale Netze (MLP) & Backpropagation.
24. **24_Model_Comparison.ipynb** – Rigoroser Metrikvergleich (F1, AUC, MCC, Cohen's Kappa).
25. **25_Hyperparameter_Tuning.ipynb** – Grid Search, Random Search & Bayes'sche Optimierung.
26. **26_Final_Model.ipynb** – Das finale Gewinnermodell & Out-of-Sample-Validierung.
27. **27_Model_Explainability.ipynb** – SHAP-Werte, Permutation Importance & LIME.
28. **28_Statistical_Validation.ipynb** – Konfidenzintervalle der Metriken & McNemar-Test.
29. **29_Final_Report.ipynb** – Wissenschaftliche Zusammenfassung & Diskussion der Entdeckungen.
30. **30_Presentation.ipynb** – Folienpräsentation der Ergebnisse für Fachtagungen.

---

## Installation und Verwendung

### 1. Conda-Umgebung erstellen (empfohlen):
```bash
conda env create -f environment.yml
conda activate maus-monitor-ml
```

### 2. Pip-Installation:
```bash
pip install -r requirements.txt
```

### 3. Starten der Notebooks:
```bash
jupyter notebook
```

---

## Lizenz

Dieses Projekt ist unter der **MIT-Lizenz** lizenziert. Siehe die Datei `LICENSE` für Details.
