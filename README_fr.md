<div align="center">
    <a href="https://github.com/gauthier-scano/draw2/blob/main/README_fr.md">[🇫🇷 Français](README_fr.md)</a>
</div>

<br>

Ce fork est une refonte du code Python utilisé pour s'interfacer avec l'IA de DRAW2.

**Ce projet se veut le plus compact et concis possible**. Son objectif : exposer, via un unique fichier, un serveur WebSocket auquel il est possible de se connecter pour envoyer des images en base64 à traiter pour la détection.

Il intègre également le nécessaire pour télécharger toutes les dépendances en local dans le cas où vous utilisez un serveur ou une application coupé d'internet.

Ce code est utilisé pour réaliser des détections de cartes sur le site du Remote Duel Arena, qui permet à n'importe quels joueurs dans le monde de faire des duels directement dans leur navigateur : <a href="https://remoteduelarena.fr">https://remoteduelarena.fr</a>.

Projet sous licence [GNU Affero General Public License v3.0](LICENCE); toutes les contributions sont les bienvenues.

Merci à HichTala pour ce merveilleux travail ! Créons des choses incroyables avec !

TODO: rendre plus friendly le soft : paramètres en entrée, vérification de l'image (taille, corruption, gestion des erreurs) pour exposition ouverte sur le monde.

---

## 🛠️ Installation

Pour installer ce fork, suivez les mêmes étapes d'installation que décrites dans le répertoire de DRAW2 :

```
git clone https://github.com/gauthier-scano/draw2
cd draw2
python -m pip install .
```

## 🚀 Usage

Une fois l'installation terminée, lancez simplement le serveur via la commande ci-dessous :

```shell
python app.py
```

**Aucune option n'est supportée**. Par défaut, le serveur WebSocket est démarré sur localhost:8765.
Vous pouvez modifier ce comportement simplement en modifiant les arguments passés à la classe App lors de son instanciation (arguments 2 and 3).

Une fois connecté au serveur WebSocket, tous les échanges sont faits en JSON.
La taille maximale autorisée d'un message WebSocket est configurée sur 10Mo.
Vous pouvez envoyer des messages de 2 types :

### 1) Traitement d'une image en base64 :

```
{
    "type": "analyze",
    "transactionId": string|integer, identifiant unique de la transaction. Tout étant traité en asynchrone, cet identifiant sera spécifié dans la réponse renvoyée par le serveur
    "data": string, image en base64, avec ou sans le mimeType (data:[...],)
}
```

La réponse sera sous la forme :

```
{
    "status": "success",
    "transactionId": string|integer, identifiant unique de la transaction
    "result": [{
        "box": array, tableau de coordonnées délimitant le forme détectée sous la forme [x1, y1, x2, y2...],
        "result": [{
            "label": string, nom de la carte détectée sous la forme [NOM]-[ID CARTE],
            "score": number, taux de fiabilité de la reconnaissance entre 0 et 1, 1 étant le meilleur possible
        }, ...]
    }, ...]
}
```

### 2) Fermeture propre de la connexion :

```
{
    "type": "close"
}
```

Aucune réponse n'est retournée.

### Gestion des erreurs :

En cas d'erreur, un objet est retourné. 3 erreurs possibles : le format JSON est incorrect, l'id de la transaction est manquant, une propriété est manquante dans l'objet `data`. **Toutes les autres erreurs doivent être traitées au niveau applicatif supérieur (image corrompue ou trop grosses typiquement).**

```
{
    "status": "error",
    "message": string, raison de l'erreur
}
```
