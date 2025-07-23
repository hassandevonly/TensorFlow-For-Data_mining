# Prédiction du prix des voitures avec TensorFlow

Ce projet a été réalisé dans le cadre du cours de **Data Mining** du Master IASD.  
Il utilise des techniques de **machine learning supervisé** via **TensorFlow** pour prédire le prix de voitures à partir de caractéristiques disponibles dans un dataset.

---
## Objectifs pédagogiques

- Comprendre le cycle de vie d’un projet de data mining (prétraitement, modélisation, évaluation)
- Mettre en œuvre un modèle de **régression avec TensorFlow**
- Comparer les performances via des métriques standards (MAE, MSE, R²)
- Appliquer les bonnes pratiques de manipulation de données avec **Pandas / NumPy**

---

## Contenu du projet

| Fichier | Description |
|--------|-------------|
| `Exercice_TensorFlow_Prix_Voiture.ipynb` | Notebook principal pour la régression TensorFlow sur les prix de voitures |
| `dataset.csv` | Données d'entraînement issues de la plateforme CarDekho |
| `HRDataset.ipynb` | Notebook d'analyse sur un dataset RH (exercice complémentaire) |
| `HRDataset_v14.csv` | Données RH utilisées par `HRDataset.ipynb` |
| `Projet TensorFlow.pptx` | Diaporama PowerPoint utilisé pour la présentation du projet |
| `TensorFlow.pdf` | Support théorique du cours sur TensorFlow (PDF) |
| `README.md` | Présentation détaillée du projet, instructions et contexte |
| `requirements.txt` | Fichier listant les dépendances Python à installer avec `pip` |
| `environment_tf_env.yml` | Environnement Conda prêt à l’emploi pour garantir la compatibilité |


---

## Modèle utilisé

- **Type de problème** : Régression
- **Framework** : TensorFlow + Keras (API haut-niveau)
- **Architecture** : `Sequential` avec 2-3 couches denses (`Dense`)
- **Fonction de perte** : Mean Squared Error (MSE)
- **Optimiseur** : Adam
- **Évaluation** : MAE, R² Score, courbes de pertes

---

## Visualisation des performances

Le modèle est évalué graphiquement à travers :

- Courbe d’évolution de la perte (`loss`)
- Comparaison `y_test vs y_pred` en scatter plot
- Affichage de la métrique R²
- Heatmap de corrélation entre variables du dataset

---

## Exécution locale ou via Colab

### 1. Sur Google Colab (recommandé)

Ouvrir le fichier `Exercice_TensorFlow_Prix_Voiture.ipynb` via :  
- https://colab.research.google.com/github/hassandevonly/TensorFlow-For-Data_mining/blob/main/Exercice_TensorFlow_Prix_Voiture.ipynb
    
    ## Chargement automatique des données
    
    Dans Google Colab ou tout environnement distant, utilisez le lien GitHub suivant pour charger directement le fichier CSV :
    
    ```python
    url = "https://raw.githubusercontent.com/hassandevonly/TensorFlow-For-Data_mining/main/dataset.csv"
    df = pd.read_csv(url)
    ```
    
    Cela vous évite d’avoir à uploader manuellement le fichier `dataset.csv`.
    
    ---

### 2. En local (Jupyter)

```bash
git clone https://github.com/hassandevonly/TensorFlow-For-Data_mining.git
cd TensorFlow-For-Data_mining
```
---

## Configuration de l’environnement (Conda recommandé)

Si vous utilisez Anaconda, vous pouvez créer un environnement compatible avec TensorFlow en ouvrant `Anaconda Prompt` et suivez ces etapes.

### Étapes :
```bash
conda env create -f environment_tf_env.yml
conda activate tf_env
pip install notebook
jupyter notebook
```

## Installer les dépendances (optionnel)
```bash
pip install -r requirements.txt
```
## Lancer le projet dans Jupyter Notebook
- Ouvre `Exercice_TensorFlow_Prix_Voiture.ipynb` et `HRDataset.ipynb` dans Jupyter Notebook.
---
## Technologies utilisées
- Python 3.x
- Pandas / NumPy
- TensorFlow / Keras
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook

## Références
- Documentation TensorFlow : [https://www.tensorflow.org](https://www.tensorflow.org/)
- Guide de l’API Keras : [https://keras.io](https://keras.io/)

## Perspectives d’amélioration
- Intégration d’un modèle avec TensorFlow Decision Forests
- Sauvegarde du modèle (model.save() et load_model())
- Interface simple avec Streamlit ou Gradio
- Étude comparative avec des modèles de scikit-learn (RandomForest, XGBoost)

# Réalisé par
- **BANGOURA Mohamed El Hassan**
- **GOZO Amé Ethiam Godwin**
- **DIALLO Rabiatou**



