# Importation des bibliothèques nécessaires
import numpy as np  # Pour manipuler les tableaux numériques
import cv2 as cv  # OpenCV pour manipuler les images
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os  # Pour parcourir les dossiers et fichiers
from classifieur_euclidien import extract_profiles  # Fonction pour extraire les profils d'une image
from collections import defaultdict  # Pour créer un dictionnaire avec des valeurs par défaut

# Paramètre global
d = 10  # Nombre de profils à extraire par image


# Fonction pour calculer la distance euclidienne entre deux vecteurs
def distance_euclidienne(vecteur1, vecteur2):
    """
    Calcule la distance euclidienne entre deux vecteurs.
    La distance euclidienne est utilisée pour mesurer la similarité entre deux vecteurs de caractéristiques.
    """
    return np.linalg.norm(vecteur1 - vecteur2)  # Norme euclidienne (distance)


# Fonction pour calculer les probabilités à partir des distances
def calculer_probabilites(classes_distances):
    """
    Prend un dictionnaire de distances pour chaque classe et renvoie un dictionnaire des probabilités associées.
    Utilise une exponentielle inversée pour transformer les distances en probabilités.
    """
    proba_classes = {}

    # Conversion des distances en tableau et aplatissement
    distances = np.array(list(classes_distances.values())).flatten()

    # Applique la fonction exponentielle inversée pour favoriser les distances faibles
    exp_neg_distances = np.exp(-distances)

    # Calcul de la somme des exponentielles
    somme_exp = np.sum(exp_neg_distances)

    # Si la somme est nulle (cas exceptionnel), ne normalise pas
    if somme_exp == 0:
        probabilites = exp_neg_distances
    else:
        # Normalisation des probabilités pour qu'elles aient une somme égale à 1
        probabilites = exp_neg_distances / somme_exp

    # Affecte les probabilités à chaque classe
    idx = 0
    for classe in classes_distances.keys():
        proba_classes[classe] = probabilites[idx]
        idx += 1

    return proba_classes  # Renvoie un dictionnaire des probabilités


# Fonction pour reconnaître la classe (caractère) d'une image
def reconnaitre_caractere(image, centres_classes):
    """
    Cette fonction prend une image et tente de reconnaître à quelle classe (caractère) elle appartient.
    Elle compare l'image avec les centres de chaque classe en utilisant la distance euclidienne.
    """
    # Extraction du vecteur de caractéristiques (profils) de l'image
    vecteur = extract_profiles(image, d)

    # Initialisation d'un dictionnaire pour stocker les distances à chaque centre de classe
    classes_distances = {}
    for classe, centre in centres_classes.items():
        classes_distances[classe] = []
        # Calcul de la distance entre le vecteur d'image et le centre de chaque classe
        dist = distance_euclidienne(vecteur, centre)
        classes_distances[classe].append(dist)

    # Conversion des distances en probabilités pour chaque classe
    probabilites = calculer_probabilites(classes_distances)

    # La classe prédite est celle avec la probabilité la plus élevée
    classe_predite = max(probabilites, key=probabilites.get)

    return probabilites, classe_predite  # Renvoie les probabilités et la classe prédite

def evaluation_base(dossier_evaluation, centres_classes):
    """
    Évalue les performances du modèle en testant un ensemble d'images du dossier d'évaluation.
    Renvoie les résultats des probabilités, le taux global de reconnaissance, le taux par classe,
    et les listes de classes réelles et prédites pour calculer la matrice de confusion.
    """
    resultats_probabilites = {}  # Stockage des résultats pour chaque image
    correct_predictions = 0  # Nombre de prédictions correctes
    total_images = 0  # Nombre total d'images évaluées
    taux_par_classe = defaultdict(lambda: {'correct': 0, 'total': 0})  # Taux par classe

    # Initialisation de la liste des classes réelles et prédites
    classes_reelles = []
    classes_predites = []

    # Parcourt chaque classe dans le dossier d'évaluation
    for nom_classe in os.listdir(dossier_evaluation):
        chemin_classe = os.path.join(dossier_evaluation, nom_classe)

        # Si ce n'est pas un dossier, ignorer
        if not os.path.isdir(chemin_classe):
            continue

        # Parcourt chaque fichier image dans la classe
        for nom_fichier in os.listdir(chemin_classe):
            resultats_probabilites[nom_fichier] = []
            chemin_image = os.path.join(chemin_classe, nom_fichier)
            image = cv.imread(chemin_image, cv.IMREAD_GRAYSCALE)  # Charge l'image en niveaux de gris

            # Si l'image ne peut pas être chargée, continuer
            if image is None:
                print(f"Erreur lors du chargement de l'image {chemin_image}")
                continue

            # Reconnaît la classe prédite et les probabilités pour cette image
            probabilites, classe_predite = reconnaitre_caractere(image, centres_classes)

            # Stocker les probabilités et la classe prédite
            resultats_probabilites[nom_fichier].append({
                'probabilites': probabilites,
                'classe_predite': classe_predite
            })

            # Ajouter les classes réelles et prédites
            classes_reelles.append(nom_classe)
            classes_predites.append(classe_predite)

            # Vérifier si la prédiction est correcte
            est_correct = (nom_classe == classe_predite)
            if est_correct:
                correct_predictions += 1  # Incrémenter les prédictions correctes
                taux_par_classe[nom_classe]['correct'] += 1  # Incrémenter correct pour la classe

            # Incrémenter le total d'images par classe et globalement
            taux_par_classe[nom_classe]['total'] += 1
            total_images += 1

    # Calcul du taux global de reconnaissance
    taux_global = correct_predictions / total_images * 100

    # Calcul du taux de reconnaissance par classe
    taux_par_classe_final = {}
    for classe, stats in taux_par_classe.items():
        taux_classe = stats['correct'] / stats['total'] * 100
        taux_par_classe_final[classe] = taux_classe

    return resultats_probabilites, taux_global, taux_par_classe_final, classes_reelles, classes_predites

def main():
    """
    Fonction principale qui charge les centres de classes, exécute l'évaluation sur un ensemble d'images,
    affiche les résultats, et les sauvegarde.
    """
    # Chargement des centres de classes à partir d'un fichier .npy
    centres_classes = np.load('centres_classes.npy', allow_pickle=True).item()

    # Chemin vers le dossier d'évaluation (ensemble de test)
    dossier_evaluation = "D:/Documents/INFO5/VA52/TP1/CCPD_Dataset_preproc/test/"

    # Appel de la fonction d'évaluation pour obtenir les résultats et les taux
    resultats, taux_global, taux_par_classe, classes_reelles, classes_predites = evaluation_base(dossier_evaluation, centres_classes)

    # Affichage des résultats de reconnaissance pour chaque image
    for name, result in resultats.items():
        probabilites = result[0]['probabilites']
        classe_predite = result[0]['classe_predite']
        print(f"Image : {name}, Classe prédite : {classe_predite}, Probas : {probabilites}\n")

    # Affichage du taux de reconnaissance par classe
    print("\nTaux de reconnaissance par classe :")
    for classe, taux in taux_par_classe.items():
        print(f"Classe {classe} : {taux:.2f}%")

    # Affichage du taux de reconnaissance global
    print(f"Taux de reconnaissance global : {taux_global:.2f}%")

    # Calcul et affichage de la matrice de confusion
    labels = list(centres_classes.keys())  # Liste des classes
    matrice_confusion = confusion_matrix(classes_reelles, classes_predites, labels=labels)

    print("\nMatrice de confusion :")
    print(matrice_confusion)

    # Affichage de la matrice de confusion sous forme graphique (heatmap)
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrice_confusion, annot=True, fmt="d", cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
    plt.title("Matrice de confusion")
    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")
    plt.show()

    # Sauvegarde des probabilités et résultats dans un fichier .npy
    np.save('probabilites_evaluation.npy', resultats)




# Exécution de la fonction principale
if __name__ == '__main__':
    main()
