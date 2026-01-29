import json
import base64

import math
import cv2
import torch
import numpy as np
import asyncio
import websockets
import shutil

from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, pipeline
from ultralytics import YOLO
from datasets import load_dataset

from huggingface_hub import hf_hub_download
from huggingface_hub import snapshot_download

@dataclass
class App:
    config_directory: str
    host: str
    port: int

    def init(self):
        with open(config_directory + "dataset.json", "rb") as f:
            self.label2id = json.load(f)

        with open(config_directory + "draw_config.json", "rb") as f:
            self.configs = json.load(f)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model_regression = YOLO(f"{config_directory}ygo_yolo.pt")

        image_processor = AutoImageProcessor.from_pretrained(
            config_directory + "vit_processor",
            use_fast=True
        )
        self.classifier = pipeline(
            "image-classification",
            model="HichTala/draw2",
            image_processor=image_processor,
            device_map=self.device
        )

    def start_server(self):
        asyncio.run(self.run_server())
        
    async def run_server(self):
        async with websockets.serve(self.handler, self.host, self.port, max_size=20*1024*1024):
            print("WebSocket started at ws://" + self.host + ":" + str(self.port))
            await asyncio.Future()

    async def handler(self, websocket):
        print("Client connected.")

        while True:
            try:
                message = await websocket.recv()

                print("Receiving message from client.")

                data = json.loads(message)

                if data.get("transactionId"):
                    if data.get("type") == "analyze" and "data" in data and "image" in data["data"]:
                        results = self.model_regression.track(
                            source=self.base64_to_image(data["data"]["image"]),
                            show_labels=False,
                            save=False,
                            device=self.device,
                            stream=True,
                            verbose=False
                        )

                        result_list = []
                        for result in results:
                            outputs = self.process(result, self.configs, False)
                            
                            for output in outputs:
                                result_list.append(output)
                        
                        response = {
                            "status": "success",
                            "transactionId": data["transactionId"],
                            "result": result_list
                        }
                    elif data.get("type") == "close":
                        await websocket.close()
                    else:
                        response = {"status": "error", "message": "Missing properties in data objet."}
                else:
                    response = {"status": "error", "message": "Property 'transactionId' is required in payload."}
            except json.JSONDecodeError:
                response = {"status": "error", "message": "Invalid JSON"}
            
            await websocket.send(json.dumps(response))
    
    def download_dependencies(self):
        Path(config_directory).mkdir(parents=True, exist_ok=True)

        # YOLO PT file
        print("Getting YOLO file.")

        yolo_path = hf_hub_download(repo_id="HichTala/draw2", filename="ygo_yolo.pt")

        print("YOLO local file path :" + str(yolo_path))

        link = Path(yolo_path)
        real_file = link.resolve(strict=True)

        shutil.move(real_file, config_directory + "ygo_yolo.pt")

        # Processor download
        print("Downloading processor")

        processor_directory = config_directory + "vit_processor"
        local_dir = snapshot_download(
            repo_id="google/vit-base-patch16-224-in21k",
            repo_type="model",
            local_dir=processor_directory,
            local_dir_use_symlinks=False
        )

        print("Processor location : " + str(local_dir))

        # datasets download, process and local save
        dataset = load_dataset("HichTala/ygoprodeck-dataset", split="train")
        labels = dataset.features["label"].names
        label2id = dict()

        for i, label in enumerate(labels):
            label2id[label] = str(i)

        with open(config_directory + "dataset.json", "w", encoding="utf-8") as f:
            json.dump(label2id, f, ensure_ascii=False, indent=4)

        # config file
        config_path = hf_hub_download(repo_id="HichTala/draw2", filename="draw_config.json")

        link = Path(config_path)
        real_file = link.resolve(strict=True)

        shutil.move(real_file, config_directory + "draw_config.json")

        print("Download done.")

    def base64_to_image(self, b64_string: str) -> np.ndarray:
        if "," in b64_string:
            b64_string = b64_string.split(",")[1]

        img_bytes = base64.b64decode(b64_string)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # BGR
        
        return img

    def process(self, result, configs, show):
        outputs = []

        if show:
            image = result.orig_img.copy()

        if result.obb.id is not None:
            for nbox, boxe in enumerate(result.obb.xyxyxyxyn):
                boxe = np.float32(
                    [[b[0] * result.orig_img.shape[1], b[1] * result.orig_img.shape[0]] for b in boxe.cpu()]
                )
                obb = np.intp(boxe)
                xy1, _, xy2, _ = obb

                output_pts = np.float32([
                    [224, 224],
                    [224, 0],
                    [0, 0],
                    [0, 224]
                ])
                perspective_transform = cv2.getPerspectiveTransform(boxe, output_pts)
                roi = cv2.warpPerspective(
                    result.orig_img, perspective_transform, (224, 224), flags=cv2.INTER_LINEAR
                )
                contours = self.extract_contours(
                    roi,
                    d=configs["bilateral_filter_d"],
                    sigma_color=configs["bilateral_filter_sigma_color"],
                    sigma_space=configs["bilateral_filter_sigma_space"],
                    thresh=configs["txt_box_contour_threshold"]
                )

                if contours != ():
                    if show:
                        cv2.drawContours(image, [obb], 0, (152, 255, 119), 2)
                        cv2.imshow("Image", image)
                        cv2.waitKey(1)
                    
                    contour = contours[np.array(list(map(cv2.contourArea, contours))).argmax()]
                    box_txt, txt_aspect_ratio = self.get_txt(contour)

                    rotation = self.get_rotation(boxes=result.obb.xywhr[nbox], box_txt=box_txt)
                    #if rotation is None:
                    #    break

                    if rotation != 0:
                        roi = cv2.rotate(roi, rotation)

                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    roi = Image.fromarray(roi)

                    output = self.classifier(roi, top_k=15)

                    outputs.append({
                        "box": [x for line in obb.tolist() for x in line],  # merge box array to an unique array [x1, y1, x2, y2...]
                        "result": output
                    })
        
        return outputs

    def test_with_local_image(self, image_path: str):
        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()

        img_base64 = base64.b64encode(img_bytes).decode("utf-8")

        results = self.model_regression.track(
            source=self.base64_to_image(img_base64),
            show_labels=False,
            save=False,
            device=self.device,
            stream=True,
            verbose=False
        )

        for result in results:
            outputs = self.process(result, self.configs, True)

            for output in outputs:
                print(output)
            
    def get_txt(self, contour):
        rect = cv2.minAreaRect(contour)
        box_txt = cv2.boxPoints(rect)
        box_txt = np.intp(box_txt)

        dx = max(box_txt[:, 0]) - min(box_txt[:, 0])
        dy = max(box_txt[:, 1]) - min(box_txt[:, 1])
        txt_aspect_ratio = max(dx, dy) / min(dx, dy)

        return box_txt, txt_aspect_ratio
    
    def extract_contours(self, roi, d, sigma_color, sigma_space, thresh):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)
        equalized = cv2.equalizeHist(gray)
        _, thresh = cv2.threshold(equalized, thresh, 255, cv2.THRESH_BINARY)

        kernel = np.ones((7, 7), np.uint8)
        edged = cv2.erode(thresh, kernel, iterations=3)
        edged = cv2.dilate(edged, kernel, iterations=3)

        contours = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        return contours
    
    def get_rotation(self, boxes, box_txt):
        w = boxes[2]
        h = boxes[3]
        angle = boxes[4] % math.pi / 2
        if min(box_txt[:, 0]) < 112:  # on élimine 90 clockwise
            if max(box_txt[:, 0]) < 112:  # on vérifie 90 anticlockwise
                if min(box_txt[:, 1]) < 112 < max(box_txt[:, 1]):
                    if (h > w and angle > math.pi / 4) or (h < w and angle < math.pi / 4):
                        rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
                    else:
                        rotation = None
                else:
                    rotation = None
            else:  # on élimine les 90
                if min(box_txt[:, 1]) < max(box_txt[:, 1]) < 112:
                    if (h > w and angle < math.pi / 4) or (h < w and angle > math.pi / 4):
                        rotation = cv2.ROTATE_180
                    else:
                        rotation = None
                elif 112 < min(box_txt[:, 1]) < max(box_txt[:, 1]):
                    if (h > w and angle < math.pi / 4) or (h < w and angle > math.pi / 4):
                        rotation = 0
                    else:
                        rotation = None
                else:
                    rotation = None
        else:  # on vérifie 90 clockwise
            if min(box_txt[:, 1]) < 112 < max(box_txt[:, 1]):
                if (h > w and angle > math.pi / 4) or (h < w and angle < math.pi / 4):
                    rotation = cv2.ROTATE_90_CLOCKWISE
                else:
                    rotation = None
            else:
                rotation = None

        return rotation

if __name__ == "__main__":
    local_directory = str(Path(__file__).resolve().parent.as_posix())
    config_directory = local_directory + "/dependencies/"

    app = App(config_directory, "localhost", 8765)

    if False:    # set to True if you need to download all resources locally (i.e. your server/application is not connected to the web)
        app.download_dependencies()
    else:
        app.init()

        if False:  # set to True if you want to test with a local image (i.e. check if dependencies download worked)
            app.test_with_local_image(local_directory + "/test/test-3.jpg")
        else:
            app.start_server()
