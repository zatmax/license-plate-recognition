# Importer les bibliothèques nécessaires
import cv2 as cv  # OpenCV pour la manipulation d'images (lecture, écriture, etc.)
import numpy as np  # NumPy pour les opérations sur des tableaux numériques
import os  # Bibliothèque pour parcourir les fichiers et les dossiers

# Définir le nombre de composantes à extraire (profil gauche + droit pour chaque ligne sélectionnée)
d = 10  # Nombre total de composantes (5 lignes pour le profil gauche, 5 lignes pour le profil droit)


# Fonction pour extraire les profils gauche et droit à partir d'une image en niveaux de gris
def extract_profiles(image, nb_composantes=10):
    # Récupérer les dimensions de l'image (hauteur et largeur)
    hauteur, largeur = image.shape

    # Listes pour stocker les valeurs du profil gauche et droit
    profil_gauche = []
    profil_droit = []

    # Sélectionner nb_composantes / 2 lignes régulièrement espacées dans l'image
    lignes = np.linspace(0, hauteur - 1, int(nb_composantes / 2), dtype=int)

    # Parcourir les lignes sélectionnées
    for ligne in lignes:
        # Extraire la ligne de pixels correspondante
        ligne_pixels = image[ligne, :]

        # Vérifier s'il y a des pixels non noirs (différents de 0) sur cette ligne
        if np.any(ligne_pixels != 0):
            # Profil gauche : position du premier pixel non noir à gauche
            gauche = np.argmax(ligne_pixels != 0)
            # Profil droit : position du premier pixel non noir à droite (en inversant la ligne)
            droit = np.argmax(ligne_pixels[::-1] != 0)
        else:
            # Si la ligne est entièrement noire, définir gauche et droit à la largeur de l'image
            gauche = largeur
            droit = largeur

        # Normaliser les positions par la largeur de l'image et ajouter au profil gauche/droit
        profil_gauche.append(gauche / largeur)
        profil_droit.append(droit / largeur)

    # Concaténer les deux profils (gauche et droit) dans un seul vecteur
    profil_concatené = np.concatenate([profil_gauche, profil_droit])
    return profil_concatené  # Retourne le vecteur des profils concaténés


# Fonction pour calculer le centre (moyenne) des vecteurs de chaque classe
def classes_center(dossier_base):
    # Dictionnaire pour stocker les vecteurs de chaque classe
    vecteurs_classes = {}

    # Parcourir les sous-dossiers (chaque sous-dossier est une classe)
    for nom_classe in os.listdir(dossier_base):
        chemin_classe = os.path.join(dossier_base, nom_classe)

        # Vérifier que le chemin correspond bien à un dossier (et non un fichier)
        if not os.path.isdir(chemin_classe):
            continue  # Si ce n'est pas un dossier, on passe à l'élément suivant

        # Créer une entrée pour cette classe dans le dictionnaire
        vecteurs_classes[nom_classe] = []

        # Parcourir tous les fichiers (images) dans le dossier de la classe
        for nom_fichier in os.listdir(chemin_classe):
            # Créer le chemin complet de l'image
            chemin_image = os.path.join(chemin_classe, nom_fichier)
            # Charger l'image en niveaux de gris
            image = cv.imread(chemin_image, cv.IMREAD_GRAYSCALE)

            # Vérifier si l'image a été correctement chargée
            if image is None:
                print(f"Erreur lors du chargement de l'image {chemin_image}")
                continue  # Passer à l'image suivante si une erreur survient

            # Extraire le vecteur de profil pour cette image
            vecteur = extract_profiles(image, d)

            # Ajouter ce vecteur au dictionnaire de la classe correspondante
            vecteurs_classes[nom_classe].append(vecteur)

    # Calculer le centre de chaque classe (moyenne des vecteurs)
    centres_classes = {}
    for classe, vecteurs in vecteurs_classes.items():
        # Calculer la moyenne des vecteurs pour cette classe et la stocker
        centres_classes[classe] = np.mean(vecteurs, axis=0)

    return centres_classes  # Retourne les centres des classes


# Fonction principale pour exécuter le programme
def main():
    # Spécifier le chemin du dossier contenant les images classées
    dossier_base = "D:/Documents/INFO5/VA52/TP1/CCPD_Dataset_preproc/train//"

    # Calculer les centres des classes en appelant la fonction classes_center
    centres_classes = classes_center(dossier_base)

    # Afficher les centres calculés pour chaque classe
    for classe, centre in centres_classes.items():
        print(f"Classe : {classe}, Centre : {centre}")

    # Sauvegarder les centres dans un fichier .npy (format NumPy)
    np.save('centres_classes.npy', centres_classes)


# Exécuter la fonction principale si le script est lancé directement
if __name__ == '__main__':
    main()
