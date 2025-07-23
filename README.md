# Utilisation de TensorFlow dans un projet de Data Mining

Ce travail a été réalisé dans le cadre d’un exposé présenté en Master 1 IASD, dans l’unité d’enseignement **Data Mining**.  
Il a pour objectif de démontrer comment exploiter la bibliothèque **TensorFlow** pour résoudre un problème d’apprentissage supervisé, en mettant en œuvre un modèle de régression sur un jeu de données réel.

Le projet comprend :
- Une mise en pratique de TensorFlow via un notebook Python (Jupyter)
- Une analyse complète du processus de data mining (prétraitement, modélisation, évaluation)
- Une présentation théorique des concepts clés de TensorFlow (à travers un PDF et un diaporama)
- Une configuration technique reproductible, utilisable en local ou via Google Colab
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
| [`Exercice_TensorFlow_Prix_Voiture.ipynb`](https://colab.research.google.com/github/hassandevonly/TensorFlow-For-Data_mining/blob/main/Exercice_TensorFlow_Prix_Voiture.ipynb) | Notebook principal pour la régression TensorFlow sur les prix de voitures (exécutable sur Colab) |
| [`dataset.csv`](https://raw.githubusercontent.com/hassandevonly/TensorFlow-For-Data_mining/main/dataset.csv) | Données d'entraînement issues de la plateforme CarDekho |
| [`HRDataset.ipynb`](https://colab.research.google.com/github/hassandevonly/TensorFlow-For-Data_mining/blob/main/HRDataset.ipynb) | Notebook d'analyse sur un dataset RH (exercice complémentaire, exécutable sur Colab) |
| [`HRDataset_v14.csv`](https://raw.githubusercontent.com/hassandevonly/TensorFlow-For-Data_mining/main/HRDataset_v14.csv) | Données RH utilisées par `HRDataset.ipynb` |
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

    ## Exécution du fichier `HRDataset.ipynb` sur Google Colab
    
    Ouvrir le fichier `HRDataset.ipynb` directement via Google Colab :  
    https://colab.research.google.com/github/hassandevonly/TensorFlow-For-Data_mining/blob/main/HRDataset.ipynb
    
    ---

    ## Chargement automatique du dataset RH
    
    Dans Google Colab ou tout environnement distant, utilisez le lien GitHub suivant pour charger directement le fichier `HRDataset_v14.csv` :
    
    ```python
    url = "https://raw.githubusercontent.com/hassandevonly/TensorFlow-For-Data_mining/main/HRDataset_v14.csv"
    df = pd.read_csv(url)
    ```


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



