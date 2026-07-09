import os
import json
from collections import defaultdict
from PIL import Image


def parse_txt(txt_path):
    images = []
    annotations = []
    categories = {}
    img_id_map = {}
    ann_id = 1
    img_id = 1
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            img_path = parts[0]
            if len(parts) == 1:
                if img_path not in img_id_map:
                    try:
                        with Image.open(img_path) as im:
                            width, height = im.size
                    except Exception as e:
                        print(f"鏃犳硶鎵撳紑鍥剧墖: {img_path}, 璺宠繃銆傞敊璇? {e}")
                        continue
                    images.append({
                        "file_name": os.path.join(
                            # os.path.basename(os.path.dirname(os.path.dirname(img_path))),  # images
                            os.path.basename(os.path.dirname(img_path)),  # seq
                            os.path.basename(img_path)  # 00000232.png
                        )
                        ,
                        "height": height,
                        "width": width,
                        "id": img_id
                    })
                    img_id_map[img_path] = img_id
                    img_id += 1
                continue
            bbox_cls_list = parts[1:]
            if img_path not in img_id_map:
                try:
                    with Image.open(img_path) as im:
                        width, height = im.size
                except Exception as e:
                    print(f"鏃犳硶鎵撳紑鍥剧墖: {img_path}, 璺宠繃銆傞敊璇? {e}")
                    continue
                images.append({
                    "file_name": os.path.join(
                        # os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(img_path)))),
                        # os.path.basename(os.path.dirname(os.path.dirname(img_path))),  # images
                        os.path.basename(os.path.dirname(img_path)),  # seq
                        os.path.basename(img_path)  # 00000232.png
                    )
                    ,
                    "height": height,
                    "width": width,
                    "id": img_id
                })
                img_id_map[img_path] = img_id
                img_id += 1
            for bbox_cls in bbox_cls_list:
                bbox_cls = bbox_cls.split(',')
                if len(bbox_cls) < 5:
                    continue
                x1, y1, x2, y2, cls = bbox_cls
                x1 = int(float(x1))
                y1 = int(float(y1))
                x2 = int(float(x2))
                y2 = int(float(y2))
                cls = int(float(cls))
                if cls not in categories:
                    categories[cls] = {"id": cls, "name": str(cls), "supercategory": "none"}
                w = x2 - x1
                h = y2 - y1
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id_map[img_path],
                    "category_id": cls,
                    "bbox": [x1, y1, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "segmentation": []
                })
                ann_id += 1
    return images, annotations, list(categories.values())


def txt2coco(txt_path, out_json):
    images, annotations, categories = parse_txt(txt_path)
    coco_dict = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(coco_dict, f, ensure_ascii=False, indent=2)
    print(f"宸茬敓鎴怌OCO鏍煎紡json: {out_json}")


if __name__ == "__main__":
    txt_path = "train_cropped.txt"
    out_json = "train_cropped_coco.json"
    txt2coco(txt_path, out_json)
    val_txt_path = "val_cropped.txt"
    val_out_json = "val_cropped_coco.json"
    txt2coco(val_txt_path, val_out_json)


