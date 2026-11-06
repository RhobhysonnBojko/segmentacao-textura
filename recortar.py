"""Gera os recortes 512x512 em tons de cinza a partir das fotos originais (pasta fotos/).
Cada foto (1200x1600) gera 9 recortes numa grade 3x3 com sobreposição."""
import glob
import os

import cv2

os.makedirs("imagens", exist_ok=True)
n = 0
for f in sorted(glob.glob("fotos/*.jpg")):
    img = cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY)   # RGB -> cinza
    h, w = img.shape
    xs = [0, (w - 512) // 2, w - 512]
    ys = [0, (h - 512) // 2, h - 512]
    for y in ys:
        for x in xs:
            cv2.imwrite(f"imagens/img_{n:02d}.png", img[y:y + 512, x:x + 512])
            n += 1
print(n, "recortes gerados")
