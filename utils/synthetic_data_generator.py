import cv2
import numpy as np
import random
import os
from pathlib import Path
import json

class SyntheticDocumentGenerator:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "annotations").mkdir(exist_ok=True)

        self.sample_words = [
            "Lorem", "ipsum", "dolor", "sit", "amet", "consectetur", 
            "adipiscing", "elit", "sed", "do", "eiusmod", "tempor",
            "incididunt", "ut", "labore", "et", "dolore", "magna", "aliqua"
        ]

    def random_text_line(self, min_words=3, max_words=8):
        """Generate a random line of text."""
        return " ".join(random.choices(self.sample_words, k=random.randint(min_words, max_words)))

    def create_document_template(self, width=400, height=600):
        """Create a document with text."""
        doc = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Title
        cv2.rectangle(doc, (0, 0), (width, 60), (230, 230, 230), -1)
        cv2.putText(doc, self.random_text_line(2, 4), (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # Body text
        y = 100
        while y < height - 50:
            text = self.random_text_line()
            font_scale = random.uniform(0.5, 0.7)
            cv2.putText(doc, text, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0),
                        thickness=1, lineType=cv2.LINE_AA)
            y += int(30 * font_scale) + 10

        return doc

    def add_perspective_distortion(self, img, max_distortion=0.3):
        """Apply random perspective distortion."""
        h, w = img.shape[:2]
        src_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        distortion = max_distortion * min(w, h)
        dst_corners = src_corners + np.random.uniform(
            -distortion, distortion, src_corners.shape).astype(np.float32)
        dst_corners[:, 0] = np.clip(dst_corners[:, 0], -w * 0.2, w * 1.2)
        dst_corners[:, 1] = np.clip(dst_corners[:, 1], -h * 0.2, h * 1.2)
        matrix = cv2.getPerspectiveTransform(src_corners, dst_corners)
        distorted = cv2.warpPerspective(img, matrix, (w, h))
        return distorted, dst_corners

    def add_to_background(self, doc_img, doc_corners, bg_size=(800, 1200)):
        """Place distorted doc on random background."""
        background = np.random.randint(200, 255, (*bg_size, 3), dtype=np.uint8)
        max_x = bg_size[1] - doc_img.shape[1]
        max_y = bg_size[0] - doc_img.shape[0]
        start_x = random.randint(0, max(0, max_x))
        start_y = random.randint(0, max(0, max_y))
        mask = (doc_img.sum(axis=2) < 750).astype(np.uint8) * 255
        end_y = min(start_y + doc_img.shape[0], bg_size[0])
        end_x = min(start_x + doc_img.shape[1], bg_size[1])
        doc_h = end_y - start_y
        doc_w = end_x - start_x
        background[start_y:end_y, start_x:end_x] = doc_img[:doc_h, :doc_w]
        adjusted_corners = doc_corners.copy()
        adjusted_corners[:, 0] += start_x
        adjusted_corners[:, 1] += start_y
        return background, adjusted_corners

    def generate_dataset(self, num_samples=1000):
        """Generate a dataset of synthetic documents with text."""
        annotations = []
        for i in range(num_samples):
            doc_width = random.randint(300, 500)
            doc_height = random.randint(400, 700)
            document = self.create_document_template(doc_width, doc_height)
            distorted_doc, corners = self.add_perspective_distortion(
                document, max_distortion=random.uniform(0.1, 0.4))
            final_img, final_corners = self.add_to_background(distorted_doc, corners)
            if random.random() > 0.5:
                noise = np.random.normal(0, 10, final_img.shape).astype(np.uint8)
                final_img = cv2.add(final_img, noise)
            if random.random() > 0.7:
                final_img = cv2.GaussianBlur(final_img, (3, 3), 0)
            img_name = f"doc_{i:06d}.jpg"
            img_path = self.output_dir / "images" / img_name
            cv2.imwrite(str(img_path), final_img)
            annotation = {
                "image": img_name,
                "corners": final_corners.tolist(),
                "width": final_img.shape[1],
                "height": final_img.shape[0]
            }
            annotations.append(annotation)
            if i % 100 == 0:
                print(f"Generated {i} samples...")
        with open(self.output_dir / "annotations" / "annotations.json", "w") as f:
            json.dump(annotations, f, indent=2)
        print(f"Dataset generated! {num_samples} images saved to {self.output_dir}")
        return annotations

def generate_dataset(root="../data/synthetic_documents_with_text", num_samples=1000, ):
    generator = SyntheticDocumentGenerator(output_dir=root)
    annotations = generator.generate_dataset(num_samples=num_samples)
    print("Sample annotation:")
    print(json.dumps(annotations[0], indent=2))