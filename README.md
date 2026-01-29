<div align="center">
    [![Licence](https://img.shields.io/pypi/l/ultralytics)](LICENSE)
    [🇫🇷 Français](README_fr.md)
</div>

<br>

This fork is a rework of the Python code used to interface the DRAW2 AI.

The project is a **one-file-ready-to-go** that allow other applications to connect to it using WebSocket procotol then send base64 image to detect the card.

It also allow you to download all the dependencies in case you are using a Server not connected to the internet.

This code is used to detect cards on the Remote Duel Arena website, allowing players to battle directly in their browser using RTC Technologies :
<a href="https://remoteduelarena.fr">https://remoteduelarena.fr</a>.

This project is licensed under the [GNU Affero General Public License v3.0](LICENCE); all contributions are welcome.

Thank you HichTala for this fantastic work ! Let's create extraordinary things with it !

TODO: making the software more user-friendly: input parameters, image verification (size, corruption, error handling) for open exposure to the world.

---

### 🛠️ Installation

To install this fork, simply follow the same steps described in the DRAW2 Repository :

```
git clone https://github.com/HichTala/draw2
cd draw2
python -m pip install .
```

### 🚀 Usage

Once the installation is done, you can start the server using this command :

```shell
python app.py
```

**No options are supported**. By default, WebSocket Server is started on localhost:8765
You can change this by modifiying the App instance creation (argument 2 and 3).

Once you are connected to the WebSocket server, all messages are in JSON format.
The maximum allowed size for a WebSocket message is set to 10MB.
You can send 2 kinds of message :

### 1) Processing base64 image :

```
{
    "type": "analyze",
    "transactionId": string|integer, unique id of the transaction. Everything is async, this id will be used in the server response
    "data": string, base64 image, with or without mimeType (data:[...],)
}
```

Response will be :

```
{
    "status": "success",
    "transactionId": string|integer, unique id given in the request
    "result": [{
        "box": array, coordinates of the detected shape [x1, y1, x2, y2...],
        "result": [{
            "label": string, name of the card as [NAME]-[CARD ID],
            "score": number, fiabilité between 0 and 1 (the bigger the better) 
        }, ...]
    }, ...]
}
```

### 2) Close the socket connection :

```
{
    "type": "close"
}
```

No response is returned.

### Error management :

If an error occured, an object is returned. 3 errors possible : JSON format is incorrect, no transaction id specified, a property is missing in object `data`. **All others error has to be checked at the applicative layer (for example, corrupted or excessively large image).**

```
{
    "status": "error",
    "message": string, error reason
}
```